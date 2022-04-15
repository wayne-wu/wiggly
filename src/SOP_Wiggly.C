#include <UT/UT_DSOVersion.h>
#include <UT/UT_Interrupt.h>
#include <UT/UT_Matrix.h>
#include <UT/UT_Array.h>
#include <UT/UT_Assert.h>
#include <GU/GU_Detail.h>
#include <GU/GU_PrimPoly.h>
#include <GU/GU_PrimPacked.h>
#include <CH/CH_LocalVariable.h>
#include <PRM/PRM_Include.h>
#include <PRM/PRM_TemplateBuilder.h>
#include <OP/OP_Operator.h>
#include <OP/OP_OperatorTable.h>
#include <OP/OP_AutoLockInputs.h>
#include <SOP/SOP_Error.h>

#include <memory>
#include <algorithm>
#include <limits.h>

#include "Eigen/Dense"

#include "Wiggly.h"
#include "SOP_Wiggly.h"
#include "SOP_Wiggly.proto.h"

using namespace HDK_Wiggly;


const UT_StringHolder SOP_Wiggly::theSOPTypeName("wigglysolver"_sh);

void
newSopOperator(OP_OperatorTable *table)
{
		OP_Operator* op;
		op = new OP_Operator(
			SOP_Wiggly::theSOPTypeName,			// Internal name
			"Wiggly Solver",			// UI name
			SOP_Wiggly::myConstructor,	// How to build the SOP
			SOP_Wiggly::buildTemplates(),	// My parameters
			2,				// Min # of sources
			2,				// Max # of sources
			nullptr,
			OP_FLAG_GENERATOR);		// Flag it as generator
		op->setIconName("wigglysolver");
		table->addOperator(op);
}

static const char* theDsFile = R"THEDSFILE(
{
        name        parameters
				groupsimple {
						name		"properties"
						label		"Object Properties"
						parm {
								name    "massdensity"      // Internal parameter name
								label   "Mass Density" // Descriptive parameter name for user interface
								type    float
								default { "1000.0" }     // Default for this parameter on new nodes
								range   { 0! 5000.0 }   // The value is prevented from going below 2 at all.
																		// The UI slider goes up to 50, but the value can go higher.
								export  all         // This makes the parameter show up in the toolbox
																		// above the viewport when it's in the node's state.
					  }
						parm {
								name    "young"      // Internal parameter name
								label   "Young Modulus" // Descriptive parameter name for user interface
								type    float
								default { "100.0" }     // Default for this parameter on new nodes
								range   { 1! 1000.0 }   // The value is prevented from going below 2 at all.
																		// The UI slider goes up to 50, but the value can go higher.
								export  all         // This makes the parameter show up in the toolbox
																		// above the viewport when it's in the node's state.
						}
						parm {
								name    "poisson"      // Internal parameter name
								label   "Poisson Ratio" // Descriptive parameter name for user interface
								type    float
								default { "0.3" }     // Default for this parameter on new nodes
								range   { 0! 1! }   // The value is prevented from going below 2 at all.
																		// The UI slider goes up to 50, but the value can go higher.
								export  all         // This makes the parameter show up in the toolbox
																		// above the viewport when it's in the node's state.
						}
				}
				groupsimple {
						name		"compute"
						label		"Compute Settings"
						parm {
								name    "alpha"      // Internal parameter name
								label   "Mass Damping" // Descriptive parameter name for user interface
								type    float
								default { "1.0" }     // Default for this parameter on new nodes
								range   { 0! 100.0 }   // The value is prevented from going below 2 at all.
																		// The UI slider goes up to 50, but the value can go higher.
								export  all         // This makes the parameter show up in the toolbox
																		// above the viewport when it's in the node's state.
						}
						parm {
								name    "beta"      // Internal parameter name
								label   "Stiffness Damping" // Descriptive parameter name for user interface
								type    float
								default { "0.001" }     // Default for this parameter on new nodes
								range   { 0! 100.0 }   // The value is prevented from going below 2 at all.
																		// The UI slider goes up to 50, but the value can go higher.
								export  all         // This makes the parameter show up in the toolbox
																		// above the viewport when it's in the node's state.
						}
						parm {
								name    "physical"      // Internal parameter name
								label   "Physicalness" // Descriptive parameter name for user interface
								type    float
								default { "0.001" }     // Default for this parameter on new nodes
								range   { 0! 1.0 }   // The value is prevented from going below 2 at all.
																		// The UI slider goes up to 50, but the value can go higher.
								export  all         // This makes the parameter show up in the toolbox
																		// above the viewport when it's in the node's state.
						}
						parm {
								name    "modesnum"      // Internal parameter name
								label   "Number of Modes" // Descriptive parameter name for user interface
								type    integer
								default { "20" }     // Default for this parameter on new nodes
								range   { 10 50 }   // The value is prevented from going below 2 at all.
																		// The UI slider goes up to 50, but the value can go higher.
								export  all         // This makes the parameter show up in the toolbox
																		// above the viewport when it's in the node's state.
						}

						parm {
								name    "gconstant"      // Internal parameter name
								label   "G Constant"     // Descriptive parameter name for user interface
								type    float
								default { "0.0" }     // Default for this parameter on new nodes
								range   { 0.0 100.0 }   // The value is prevented from going below 2 at all.
																		// The UI slider goes up to 50, but the value can go higher.
								export  all         // This makes the parameter show up in the toolbox
																		// above the viewport when it's in the node's state.
						}
						parm {
								name    "epsilon"      // Internal parameter name
								label   "Epsilon"     // Descriptive parameter name for user interface
								type    float
								default { "1e-7" }     // Default for this parameter on new nodes
								range   { 1e-12 1e-3 }   // The value is prevented from going below 2 at all.
																		// The UI slider goes up to 50, but the value can go higher.
								export  all         // This makes the parameter show up in the toolbox
																		// above the viewport when it's in the node's state.
						}
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
	exint topologyDataId1;
	exint topologyDataId2;
	exint primitiveListDataId1;
	exint primitiveListDataId2;
	exint metaCacheCount1;
	exint metaCacheCound2;

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
		auto&& sopparms = cookparms.parms<SOP_WigglyParms>();
		auto sopcache = (SOP_WigglyCache*)cookparms.cache();
		
		GU_Detail* detail = cookparms.gdh().gdpNC();    // rest geometry to be modified
		const GU_Detail* bgdp = cookparms.inputGeo(1);  // constraints

		if (bgdp->getNumPrimitives() < 2)
		{
			cookparms.sopAddError(SOP_ERR_INVALID_SRC, "Need at least two keyframes.");
			return;
		}

		bool preComputeNeeded = true;
		bool computeNeeded = false;

		if (sopcache->wigglyObj)
			if (sopcache->prevInput1Id == detail->getUniqueId() &&
				sopcache->primitiveListDataId1 == detail->getPrimitiveList().getDataId() &&
				sopcache->topologyDataId1 == detail->getTopology().getDataId())
				preComputeNeeded = false;

		if (preComputeNeeded)
		{
			// TODO: Check if the parameter has changed too

			// If the mesh has changed then we need to recompute everything
			computeNeeded = true;

			// Clear cache data
			sopcache->wigglyObj.reset();

			WigglyParms parms;
			parms.alpha = sopparms.getAlpha();
			parms.beta = sopparms.getBeta();
			parms.d = std::min(sopparms.getModesnum(), detail->getNumPoints() * 3);
			parms.g = sopparms.getGconstant();
			parms.young = sopparms.getYoung();
			parms.eps = sopparms.getEpsilon();
			parms.p = sopparms.getMassdensity();
			parms.poisson = sopparms.getPoisson();
			parms.physical = sopparms.getPhysical();

			sopcache->wigglyObj = std::make_unique<Wiggly>(detail, parms);
			sopcache->wigglyObj->preCompute();

			sopcache->prevInput1Id = detail->getUniqueId();
			sopcache->primitiveListDataId1 = detail->getPrimitiveList().getDataId();
			sopcache->topologyDataId1 = detail->getTopology().getDataId();
			sopcache->metaCacheCount1 = detail->getMetaCacheCount();
		}
		else if (sopcache->prevInput2Id != bgdp->getUniqueId() ||
			sopcache->primitiveListDataId2 != bgdp->getPrimitiveList().getDataId() ||
			sopcache->topologyDataId2 != bgdp->getTopology().getDataId())
			computeNeeded = true;

		if (computeNeeded)
		{
			// GET THE KEYFRAMES DATA FROM SECOND INPUT
			GA_ROHandleI f_h(bgdp, GA_ATTRIB_POINT, "frame");

			Keyframes& keyframes = sopcache->wigglyObj->getKeyframes();
			keyframes.clear();

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

				keyframe.u = std::vector<UT_Vector3D>(packedDetail->getNumPoints());

				GA_Offset ptoff;
				GA_FOR_ALL_PTOFF(packedDetail, ptoff)
				{
					// Store all the keyframes
					int originalPtId = pt_h.get(ptoff);
					int packedPtId = packedDetail->pointIndex(ptoff);
					keyframe.points[originalPtId] = packedPtId;

					// Calculate the displacement
					keyframe.u[packedPtId] = packedDetail->getPos3(ptoff) - detail->getPos3(detail->pointOffset(originalPtId));
				}

				keyframe.detail = packedDetail;

				// NOTE: Assuming there's no duplicate right now
				keyframes.push_back(keyframe);
			}

			std::sort(keyframes.begin(), keyframes.end());

			// Validate keyframes
			if (!keyframes.front().hasVel || !keyframes.back().hasVel)
			{
				cookparms.sopAddError(SOP_ERR_INVALID_SRC, "Start and end keyframes must have velocity.");
				return;
			}

			sopcache->wigglyObj->compute();

			sopcache->prevInput2Id = bgdp->getUniqueId();
			sopcache->primitiveListDataId2 = bgdp->getPrimitiveList().getDataId();
			sopcache->topologyDataId2 = bgdp->getTopology().getDataId();
			sopcache->metaCacheCound2 = bgdp->getMetaCacheCount();
		}

		// Loop through all points and modify the value
		CH_Manager* chman = OPgetDirector()->getChannelManager();
		fpreal f = chman->getSample(cookparms.getCookTime());

		VecX uPos = sopcache->wigglyObj->u(f);

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

