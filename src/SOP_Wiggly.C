#include <UT/UT_DSOVersion.h>
#include <UT/UT_Interrupt.h>
#include <UT/UT_Matrix.h>
#include <UT/UT_Array.h>
#include <GU/GU_Detail.h>
#include <GU/GU_PrimPoly.h>
#include <GU/GU_PrimPacked.h>
#include <CH/CH_LocalVariable.h>
#include <PRM/PRM_Include.h>
#include <PRM/PRM_TemplateBuilder.h>
#include <OP/OP_Operator.h>
#include <OP/OP_OperatorTable.h>
#include <OP/OP_AutoLockInputs.h>

#include <memory>
#include <algorithm>
#include <limits.h>

#include "Eigen/Dense"

#include "Wiggly.h"
#include "SOP_Wiggly.h"
#include "SOP_Wiggly.proto.h"

using namespace HDK_Wiggly;


const UT_StringHolder SOP_Wiggly::theSOPTypeName("hdk_wiggly"_sh);

void
newSopOperator(OP_OperatorTable *table)
{
    table->addOperator(
	    new OP_Operator(
					 SOP_Wiggly::theSOPTypeName,			// Internal name
			     "Wiggly",			// UI name
			     SOP_Wiggly::myConstructor,	// How to build the SOP
			     SOP_Wiggly::buildTemplates(),	// My parameters
			     2,				// Min # of sources
			     2,				// Max # of sources
					 nullptr,
			     OP_FLAG_GENERATOR)		// Flag it as generator
	    );
}

static const char* theDsFile = R"THEDSFILE(
{
        name        parameters
        parm {
            name    "massmul"      // Internal parameter name
            label   "Mass Multiplier" // Descriptive parameter name for user interface
            type    float
            default { "1.0" }     // Default for this parameter on new nodes
            range   { 0! 100.0 }   // The value is prevented from going below 2 at all.
                                // The UI slider goes up to 50, but the value can go higher.
            export  all         // This makes the parameter show up in the toolbox
                                // above the viewport when it's in the node's state.
       }
       parm {
            name    "stiffnessmul"      // Internal parameter name
            label   "Stiffness Multiplier" // Descriptive parameter name for user interface
            type    float
            default { "1.0" }     // Default for this parameter on new nodes
            range   { 0! 100.0 }   // The value is prevented from going below 2 at all.
                                // The UI slider goes up to 50, but the value can go higher.
            export  all         // This makes the parameter show up in the toolbox
                                // above the viewport when it's in the node's state.
       }
       parm {
            name    "modesnum"      // Internal parameter name
            label   "Number of Modes" // Descriptive parameter name for user interface
            type    integer
            default { "20" }     // Default for this parameter on new nodes
            range   { 10! 50 }   // The value is prevented from going below 2 at all.
                                // The UI slider goes up to 50, but the value can go higher.
            export  all         // This makes the parameter show up in the toolbox
                                // above the viewport when it's in the node's state.
       }

}
)THEDSFILE";

PRM_Template*
SOP_Wiggly::buildTemplates()
{
	static PRM_TemplateBuilder templ("SOP_Wiggly.C"_sh, theDsFile);
	return templ.templates();
}

class SOP_WigglyCache : public SOP_NodeCache
{
public:
	SOP_WigglyCache() : SOP_NodeCache() {}
	~SOP_WigglyCache() override {}

	exint prevInput1Id;
	exint prevInput2Id;

	UT_UniquePtr<Wiggly> wigglyObj;
};

class SOP_WigglyVerb : public SOP_NodeVerb
{
public:
	SOP_WigglyVerb() {}
	virtual ~SOP_WigglyVerb() {}

	virtual SOP_NodeParms* allocParms() const { return new SOP_WigglyParms(); }
	virtual SOP_NodeCache* allocCache() const { return new SOP_WigglyCache(); }
	virtual UT_StringHolder name() const { return SOP_Wiggly::theSOPTypeName; }

	virtual SOP_NodeVerb::CookMode cookMode(const SOP_NodeParms* parms) const { 
		return SOP_NodeVerb::COOK_DUPLICATE; }

	virtual void cook(const SOP_NodeVerb::CookParms& cookparms) const;

	static const SOP_NodeVerb::Register<SOP_WigglyVerb> theVerb;
};

const SOP_NodeVerb::Register<SOP_WigglyVerb> SOP_WigglyVerb::theVerb;

const SOP_NodeVerb* 
SOP_Wiggly::cookVerb() const
{
	return SOP_WigglyVerb::theVerb.get();
}


void
SOP_WigglyVerb::cook(const SOP_NodeVerb::CookParms& cookparms) const
{
		// fpreal		 now = context.getTime();
		auto&& sopparms = cookparms.parms<SOP_WigglyParms>();
		auto sopcache = (SOP_WigglyCache*)cookparms.cache();
		
		GU_Detail* detail = cookparms.gdh().gdpNC();
		const GU_Detail* bgdp = cookparms.inputGeo(1);  // constraints

		CH_Manager* chman = OPgetDirector()->getChannelManager();

		fpreal f = chman->getSample(cookparms.getCookTime());

		if (detail->getUniqueId() != sopcache->prevInput1Id || bgdp->getUniqueId() != sopcache->prevInput2Id)
		{
			// Clear cache data
			sopcache->wigglyObj.reset();

			// GET THE KEYFRAMES DATA FROM SECOND INPUT
			GA_ROHandleI f_h(bgdp, GA_ATTRIB_POINT, "frame");

			Keyframes keyframes;

			for (GA_Iterator it(bgdp->getPrimitiveRange()); !it.atEnd(); ++it)
			{
				Keyframe keyframe;
				keyframe.frame = f_h.get(*it);

				const GU_PrimPacked* packedPrim = (const GU_PrimPacked*)bgdp->getPrimitive(*it);
				const GU_Detail* packedDetail = packedPrim->getPackedDetail().gdp();

				keyframe.points = std::vector<GA_Index>(detail->getNumPoints(), -1);

				keyframe.hasPos = true;
				keyframe.hasVel = packedDetail->findAttribute(GA_ATTRIB_POINT, "v") != nullptr;
				
				GA_ROHandleI pt_h(packedDetail, GA_ATTRIB_POINT, "original");

				GA_Offset ptoff;
				GA_FOR_ALL_PTOFF(packedDetail, ptoff)
				{
					// Store all the keyframes
					int originalPtId = pt_h.get(ptoff);
					keyframe.points[originalPtId] = packedDetail->pointIndex(ptoff);
				}

				keyframe.detail = packedDetail;

				// NOTE: Assuming there's no duplicate right now
				keyframes.push_back(keyframe);
			}

			// TODO: Validate keyframes

			std::sort(keyframes.begin(), keyframes.end());

			WigglyParms parms;
			parms.alpha = sopparms.getMassmul();
			parms.beta = sopparms.getStiffnessmul();
			parms.dim = sopparms.getModesnum();

			sopcache->wigglyObj = std::make_unique<Wiggly>(detail, keyframes, parms);
			// sopcache->wigglyObj->compute();

			sopcache->prevInput1Id = detail->getUniqueId();
			sopcache->prevInput2Id = detail->getUniqueId();
		}

		// Loop through all points and modify the value

		Eigen::VectorXf uPos = sopcache->wigglyObj->u(f);

		GA_Offset ptoff;
		GA_FOR_ALL_PTOFF(detail, ptoff)
		{
			UT_Vector3 p = detail->getPos3(ptoff);

			int ptidx = 3 * detail->pointIndex(ptoff);
			p += UT_Vector3(uPos(ptidx), uPos(ptidx + 1), uPos(ptidx + 2));

			detail->setPos3(ptoff, p);
		}

		detail->getP()->bumpDataId();
}

