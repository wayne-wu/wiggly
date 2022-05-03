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
#include <string>

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
								default { "30" }
								range   { 10 50 }
						}
						parm {
								name    "epsilon"
								label   "Error Tolerance"
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
								label   "Show Constraints"
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

typedef UT_SharedPtr<Wiggly> WigglyPtr;

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

	std::vector<WigglyPtr> wigglies;

	MatX K;
	MatX M;

	SOP_WigglyParms parms;
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
		
		GU_Detail* detail = cookparms.gdh().gdpNC();    // a copy of input geometry to be modified
		const GU_Detail* ogdp = cookparms.inputGeo(0);  // input rest geometry
		const GU_Detail* agdp = cookparms.inputGeo(1);  // wiggly splines

		if (detail->getNumPoints() == 0)
		{
			cookparms.sopAddError(SOP_MESSAGE, "Empty rest geometry.");
			return;
		}

		if (agdp->getNumPrimitives() == 0)
		{
			cookparms.sopAddError(SOP_MESSAGE, "No spline specified.");
			return;
		}

		bool preComputeNeeded = true;
		bool computeNeeded = false;

		if (sopcache->wigglies.size() > 0)
			if (sopcache->prevInput1Id == detail->getUniqueId() &&
				sopcache->primitiveListDataId1 == detail->getPrimitiveList().getDataId() &&
				sopcache->topologyDataId1 == detail->getTopology().getDataId())
				if (sopcache->parms.getPoisson() == sopparms.getPoisson() &&
					sopcache->parms.getYoung() == sopparms.getYoung() &&
					sopcache->parms.getMassdensity() == sopparms.getMassdensity())
					preComputeNeeded = false;

		if (preComputeNeeded)
		{
			// If the mesh has changed then we need to recompute everything
			computeNeeded = true;

			int dof = 3 * detail->getNumPoints();

			sopcache->K = MatX::Zero(dof, dof);
			sopcache->M = MatX::Zero(dof, dof);

			Wiggly::calculateMK_FEM(
				sopcache->K, sopcache->M, detail,
				sopparms.getYoung(), sopparms.getPoisson(), sopparms.getMassdensity());

			sopcache->prevInput1Id = detail->getUniqueId();
			sopcache->primitiveListDataId1 = detail->getPrimitiveList().getDataId();
			sopcache->topologyDataId1 = detail->getTopology().getDataId();
			sopcache->metaCacheCount1 = detail->getMetaCacheCount();
		}
		else if (sopcache->prevInput2Id != agdp->getUniqueId() ||
			sopcache->primitiveListDataId2 != agdp->getPrimitiveList().getDataId() ||
			sopcache->topologyDataId2 != agdp->getTopology().getDataId() ||
			sopcache->parms.getAlpha() != sopparms.getAlpha() ||
			sopcache->parms.getBeta() != sopparms.getBeta() ||
			sopcache->parms.getPhysical() != sopparms.getPhysical() ||
			sopcache->parms.getModesnum() != sopparms.getModesnum())
			computeNeeded = true;

		sopcache->parms = sopparms;

		if (computeNeeded)
		{
			WigglyParms parms;
			parms.alpha = sopparms.getAlpha();
			parms.beta = sopparms.getBeta();
			parms.d = std::min(sopparms.getModesnum(), detail->getNumPoints() * 3);
			parms.eps = sopparms.getEpsilon();
			parms.physical = sopparms.getPhysical();

			GOP_Manager groupManager;

			GA_RWHandleV3D v_h(detail->addFloatTuple(GA_ATTRIB_POINT, "v", 3));
			GA_RWHandleI og_h(detail->addIntTuple(GA_ATTRIB_POINT, "original", 1));

			GA_ROHandleF start_h(agdp, GA_ATTRIB_PRIMITIVE, "start");
			GA_ROHandleS group_h(agdp, GA_ATTRIB_PRIMITIVE, "group");

			WigglyPtr lastWiggly = nullptr;  // last wiggly spline

			sopcache->wigglies.clear();

			int splineNum = 1;

			// LOOP THROUGH EACH SPLINE
			for (GA_Iterator splineIt(agdp->getPrimitiveRange()); !splineIt.atEnd(); ++splineIt)
			{
				const GU_PrimPacked* packedSpline = (const GU_PrimPacked*)agdp->getPrimitive(*splineIt);
				if (packedSpline == nullptr)
				{
					cookparms.sopAddError(SOP_MESSAGE, "Invalid spline. Use Wiggly Spline SOP.");
					return;
				}
				
				const GU_Detail* bgdp = packedSpline->getPackedDetail().gdp();
				if (bgdp->getNumPrimitives() < 2)
				{
					cookparms.sopAddError(SOP_MESSAGE, "Need at least two constraints.");
					return;
				}

				std::string splineName = "Spline " + std::to_string(splineNum++);

				if (lastWiggly != nullptr)
				{
					// NOTE: Move the geometry to the last frame of the last spline
					// This is to ensure that the calculation of the current spline 
					// is adjusted to account for the last spline

					VecX& uPos = lastWiggly->getUEnd();
					VecX& uVel = lastWiggly->getUDotEnd();

					GA_Offset ptoff;
					GA_FOR_ALL_PTOFF(detail, ptoff)
					{
						// We are essentially adding all the necessary attributes to 
						// make detail a Wiggly Constraint.

						int ogPt = detail->pointIndex(ptoff);
						og_h.set(ptoff, ogPt);

						int uIdx = 3 * lastWiggly->groupIdx[ogPt];

						if (uIdx < 0) continue;

						detail->setPos3(ptoff, detail->getPos3(ptoff) +
							UT_Vector3(uPos(uIdx), uPos(uIdx + 1), uPos(uIdx + 2)));

						v_h.set(ptoff, UT_Vector3D(uVel(uIdx), uVel(uIdx + 1), uVel(uIdx + 2)));
					}
				}

				WigglyPtr wiggly = UTmakeShared<Wiggly>(detail, parms);

				wiggly->ptRange = detail->getPointRange();

				// Find pinned group
				const UT_StringHolder& groupPattern = group_h.get(*splineIt);
				if (groupPattern.isstring())
				{
					const GA_PointGroup* pinGroup = groupManager.parsePointGroups(
						groupPattern, GOP_Manager::GroupCreator(detail));
					wiggly->ptRange = GA_Range(*pinGroup, /*invert*/ true);
				}

				IndexMap groupIdx = IndexMap(detail->getNumPoints(), -1);
				UT_Set<GA_Index> unconstrainedPts;

				int n = 0;
				for (GA_Offset ptOff : wiggly->ptRange) {
					GA_Index ptIdx = detail->pointIndex(ptOff);
					groupIdx[ptIdx] = n++;
					unconstrainedPts.insert(ptIdx);
				}

				{
					std::string message = splineName + " - Computing eigen modes and eigen values.";
					UT_AutoInterrupt progress(message.c_str());
					if (progress.wasInterrupted())
						return;

					wiggly->setGroupIdx(groupIdx);
					wiggly->preCompute(sopcache->K, sopcache->M);

					if (progress.wasInterrupted())
						return;
				}

				GA_ROHandleI f_h(bgdp, GA_ATTRIB_POINT, "frame");

				Keyframes& keyframes = wiggly->getKeyframes();
				keyframes.clear();

				// LOOP THROUGH EACH CONSTRAINT
				for (GA_Iterator it(bgdp->getPrimitiveRange()); !it.atEnd(); ++it)
				{
					Keyframe keyframe;
					keyframe.frame = f_h.get(*it);

					const GU_PrimPacked* packedConstraint = (const GU_PrimPacked*)bgdp->getPrimitive(*it);
					if (packedConstraint == nullptr)
					{
						cookparms.sopAddError(SOP_ERR_INVALID_SRC, "Invalid constraint. Use Wiggly Constraint SOP.");
						return;
					}

					// NOTE: If it's the first spline, we use the first keyframe as specified, 
					// otherwise we will use the geometry with the last spline's data applied as the constraint
					// TODO: Does that mean u for the first keyframe will always be 0?
					const GU_Detail* packedDetail = keyframe.frame == start_h.get(*splineIt) && lastWiggly != nullptr ?
						detail : packedConstraint->getPackedDetail().gdp();

					keyframe.hasPos = true;
					keyframe.hasVel = packedDetail->findAttribute(GA_ATTRIB_POINT, "v") != nullptr;

					GA_ROHandleI pt_h(packedDetail, GA_ATTRIB_POINT, "original");

					keyframe.u = VecX::Zero(3 * unconstrainedPts.size());

					GA_Offset ptoff;
					GA_FOR_ALL_PTOFF(packedDetail, ptoff)
					{
						// Store all the keyframes
						int ogPt = pt_h.get(ptoff);

						// If the point is constrained, we do not store it
						if (unconstrainedPts.find(ogPt) == unconstrainedPts.end())
							continue;

						keyframe.range.append(ptoff);
						int uIdx = 3 * groupIdx[ogPt];
						UT_Vector3 u = packedDetail->getPos3(ptoff) - detail->getPos3(detail->pointOffset(ogPt));
						keyframe.u(uIdx) = u.x();
						keyframe.u(uIdx + 1) = u.y();
						keyframe.u(uIdx + 2) = u.z();
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

				wiggly->setFrameRange(keyframes.front().frame, keyframes.back().frame);

				{
					std::string message = splineName + " - Computing wiggly spline coefficients.";
					UT_AutoInterrupt progress(message.c_str());
					if (progress.wasInterrupted())
						return;

					int err = wiggly->compute(progress);
					if (err > 0)
					{
						cookparms.sopAddError(SOP_MESSAGE, "Wiggly computation failed. Check your parameters.");
						return;
					}
				}

				sopcache->wigglies.push_back(wiggly);

				lastWiggly = wiggly;
			}

			sopcache->prevInput2Id = agdp->getUniqueId();
			sopcache->primitiveListDataId2 = agdp->getPrimitiveList().getDataId();
			sopcache->topologyDataId2 = agdp->getTopology().getDataId();
			sopcache->metaCacheCound2 = agdp->getMetaCacheCount();
		}

		// Loop through all points and modify the value
		CH_Manager* chman = OPgetDirector()->getChannelManager();
		fpreal f = chman->getSample(cookparms.getCookTime());

		// Find the current wiggly spline
		int idx = -1;

		for (int i = 0; i < sopcache->wigglies.size(); ++i)
			if (sopcache->wigglies[i]->isInFrameRange(f))
			{
				idx = i;
				break;
			}

		if (idx < 0)
		{
			cookparms.sopAddWarning(SOP_MESSAGE, "No wiggly spline found for the current frame.");
			return;
		}

		Wiggly& wiggly = *sopcache->wigglies[idx];

		VecX uPos = wiggly.u(f);

		GA_Offset ptoff;
		GA_FOR_ALL_PTOFF(detail, ptoff)
		{
			int ogPt = detail->pointIndex(ptoff);

			UT_Vector3 u(0,0,0);

			// Apply displacement from all the previous splines
			for (int i = 0; i <= idx; ++i)
			{
				VecX& lastPos = i == idx ? uPos : sopcache->wigglies[i]->getUEnd();

				int uIdx = 3 * sopcache->wigglies[i]->groupIdx[ogPt];

				if (uIdx < 0) continue;

				u += UT_Vector3(lastPos(uIdx), lastPos(uIdx + 1), lastPos(uIdx + 2));
			}

			detail->setPos3(ptoff, ogdp->getPos3(ptoff) + u);
		}

		detail->getP()->bumpDataId();
}

