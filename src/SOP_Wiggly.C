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
						label		"Material Properties"
						parm {
								name    "massdensity"
								label   "Mass Density"
								type    float
								default { "1.0" }    
								range   { 0! 100.0 } 
					  }
						parm {
								name    "young"     
								label   "Stiffness" 
								type    float
								default { "75.0" }    
								range   { 1! 1000.0 }  
						}
						parm {
								name    "poisson"      
								label   "Poisson Ratio"
								type    float
								default { "0.45" }
								range   { 0! 0.499! }
						}
				}
				groupsimple {
						name		"compute"
						label		"Compute Settings"
						parm {
			        name    "pingroup"
			        cppname "PinGroup"
			        label   "Pin Constraint Group"
			        type    string
			        default { "" }
			        parmtag { "script_action" "import soputils\nkwargs['geometrytype'] = (hou.geometryType.Points,)\nkwargs['inputindex'] = 0\nsoputils.selectGroupParm(kwargs)" }
			        parmtag { "script_action_help" "Select geometry from an available viewport.\nShift-click to turn on Select Groups." }
			        parmtag { "script_action_icon" "BUTTONS_reselect" }
				    }
						parm {
								name    "alpha"
								label   "Mass Damping"
								type    float
								default { "0.05" } 
								range   { 0! 1.0 }
						}
						parm {
								name    "beta"
								label   "Stiffness Damping"
								type    float
								default { "0.005" }
								range   { 0! 0.01 }
						}
						parm {
								name    "physical"
								label   "Dynamics"
								type    float
								default { "0.001" }
								range   { 0! 1.0 }
						}
						parm {
								name    "modesnum"
								label   "Number of Modes"
								type    integer
								default { "20" }
								range   { 10 50 }
						}
						parm {
								name    "gconstant"
								label   "G Constant"
								type    vector
							  size		3
								default { "0" "0" "0" }
						}
						parm {
								name    "epsilon"
								label   "Epsilon"
								type    float
								default { "1e-7" }
								range   { 1e-12 1e-3 }
						}
				}
				groupsimple {
						name		"visualization"
						label		"Visualization"
						parm {
								name    "guidegeo"
								label   "Show Guide Geometry"
								type    toggle
								default { "1" }    
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
	SOP_WigglyParms wigglyParms;
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
		const SOP_WigglyParms& sopparms = cookparms.parms<SOP_WigglyParms>();
		auto sopcache = (SOP_WigglyCache*)cookparms.cache();
		
		GU_Detail* detail = cookparms.gdh().gdpNC();    // rest geometry to be modified
		const GU_Detail* bgdp = cookparms.inputGeo(1);  // constraints

		if (bgdp->getNumPrimitives() < 2)
		{
			cookparms.sopAddError(SOP_ERR_INVALID_SRC, "Need at least two constraints.");
			return;
		}

		GOP_Manager groupManager;
		GA_Range ptRange = detail->getPointRange();

		const UT_StringHolder& groupPattern = sopparms.getPinGroup();
		if (groupPattern.isstring())
		{
			const GA_PointGroup* pinGroup = groupManager.parsePointGroups(
				groupPattern, GOP_Manager::GroupCreator(detail));
			ptRange = GA_Range(*pinGroup, true);
		}

		// groupIdx will map from originalPtIdx to constrainedIdx
		IndexMap groupIdx = IndexMap(detail->getNumPoints(), -1);
		UT_Set<GA_Index> unconstrainedPts;

		int n = 0;
		for (GA_Offset ptOff : ptRange) {
			GA_Index ptIdx = detail->pointIndex(ptOff);
			groupIdx[ptIdx] = n++;
			unconstrainedPts.insert(ptIdx);
		}

		bool preComputeNeeded = true;
		bool computeNeeded = false;

		if (sopcache->wigglyObj)
			if (sopcache->prevInput1Id == detail->getUniqueId() &&
				sopcache->primitiveListDataId1 == detail->getPrimitiveList().getDataId() &&
				sopcache->topologyDataId1 == detail->getTopology().getDataId())
				if (sopcache->wigglyParms == sopparms)
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
			sopcache->wigglyObj->setGroupIdx(groupIdx);
			sopcache->wigglyObj->preCompute();

			// TODO: Find a better way to store the parameters.
			// Maybe we can pass the sopparms object directly to wiggly
			sopcache->wigglyParms = sopparms;

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
				if (packedPrim == nullptr)
				{
					cookparms.sopAddError(SOP_ERR_INVALID_SRC, "Invalid constraints. Use Wiggly Constraint SOP.");
					return;
				}

				const GU_Detail* packedDetail = packedPrim->getPackedDetail().gdp();

				keyframe.hasPos = true;
				keyframe.hasVel = packedDetail->findAttribute(GA_ATTRIB_POINT, "v") != nullptr;

				GA_ROHandleI pt_h(packedDetail, GA_ATTRIB_POINT, "original");

				keyframe.u = std::vector<UT_Vector3D>(n);

				GA_Offset ptoff;
				GA_FOR_ALL_PTOFF(packedDetail, ptoff)
				{
					// Store all the keyframes
					int ogPt = pt_h.get(ptoff);

					// If the point is constrained, we do not store it
					if (unconstrainedPts.find(ogPt) == unconstrainedPts.end())
						continue;

					keyframe.range.append(ptoff);
					keyframe.u[groupIdx[ogPt]] = packedDetail->getPos3(ptoff) - detail->getPos3(detail->pointOffset(ogPt));
				}

				keyframe.detail = packedDetail;

				// NOTE: Assuming there's no duplicate right now
				keyframes.push_back(keyframe);
			}

			std::sort(keyframes.begin(), keyframes.end());

			// Validate keyframes
			if (!keyframes.front().hasVel || !keyframes.back().hasVel)
			{
				cookparms.sopAddError(SOP_MESSAGE, "Start and end keyframes must have velocity.");
				return;
			}

			int err = sopcache->wigglyObj->compute();
			if (err > 0)
			{
				cookparms.sopAddError(SOP_MESSAGE, "Computation failed. Check your parameters.");
				return;
			}

			sopcache->prevInput2Id = bgdp->getUniqueId();
			sopcache->primitiveListDataId2 = bgdp->getPrimitiveList().getDataId();
			sopcache->topologyDataId2 = bgdp->getTopology().getDataId();
			sopcache->metaCacheCound2 = bgdp->getMetaCacheCount();
		}

		// Loop through all points and modify the value
		CH_Manager* chman = OPgetDirector()->getChannelManager();
		fpreal f = chman->getSample(cookparms.getCookTime());

		VecX uPos = sopcache->wigglyObj->u(f);

		for(GA_Offset ptoff : ptRange)
		{
			UT_Vector3 p = detail->getPos3(ptoff);

			int uIdx = 3 * groupIdx[detail->pointIndex(ptoff)];
			p += UT_Vector3(uPos(uIdx), uPos(uIdx + 1), uPos(uIdx + 2));

			detail->setPos3(ptoff, p);
		}

		detail->getP()->bumpDataId();
}

