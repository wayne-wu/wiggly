#pragma once

#include <vector>
#include <utility>
#include <unordered_map>

#include <UT/UT_Vector3.h>
#include <UT/UT_Array.h>
#include <UT/UT_ThreadedAlgorithm.h>
#include <GU/GU_Detail.h>

#include "Eigen/Dense"

// NOTE: Not sure if we should include Houdini dependencies here.
// On one hand, it would save memory to not have to duplicate data,
// but then you can't decouple the system anymore


namespace HDK_Wiggly {

	typedef double scalar;  // <------- Set precision here
	typedef Eigen::VectorXd VecX;
	typedef Eigen::MatrixXd MatX;
	
	typedef std::vector<int> IndexMap;

	struct Keyframe
	{
		float t;
		float frame;
		GA_OffsetArray range;
		VecX u;
		bool hasPos;
		bool hasVel;
		const GU_Detail* detail;

		bool operator< (const Keyframe& other) { return frame < other.frame; }
	};

	typedef std::vector<Keyframe> Keyframes;

	/*
	Parameters that are passed from the SOP node.
	*/
	struct WigglyParms
	{
		scalar alpha;
		scalar beta;
		UT_Vector3D g;
		int d = 20;
		scalar cA = 100.0;
		scalar cB = 100.0;
		scalar p = 1000;
		scalar young;
		scalar eps = 1e-6;
		scalar poisson;
		scalar physical;
	};

	class Wiggly
	{

	public:
		Wiggly(const GU_Detail* mgdp, const WigglyParms& sopparms)
			: mesh(mgdp), parms(sopparms) { }
		~Wiggly() {}

		int compute(UT_AutoInterrupt& progress);
		void preCompute(const MatX& K_full, const MatX& M_full);

		VecX u(const float f);
		VecX uDot(const float f);
		
		scalar totalEnergy(const VecX& c);
		scalar integrand(const float t, const int d, const scalar delta, const scalar lambda, const VecX& coeffs);
		int getNumPoints() { return mesh->getNumPoints(); }
		Keyframes& getKeyframes() { return keyframes; }
		WigglyParms& getParms() { return parms; }

		int getConstraintIdx(const int ogIdx) { return groupIdx[ogIdx]; }
		void setGroupIdx(const IndexMap& arr) { groupIdx = arr; }

		bool isInFrameRange(const float f) { return startFrame <= f && endFrame >= f; }
		void setFrameRange(const float start, const float end) { startFrame = start; endFrame = end; }
		float getEndFrame() { return endFrame;  }

		VecX& getUEnd() { return uEnd; }
		VecX& getUDotEnd() { return uDotEnd; }

		static void calculateMK_FEM(MatX& K, MatX& M, const GU_Detail* mesh, const scalar E, const scalar v, const scalar p);

		// groupIdx will map from originalPtIdx to constrainedIdx
		IndexMap groupIdx;

		// A range of points that should be looped over
		GA_Range ptRange;

	protected:

		// Basis Functions
		scalar b(const float t, const scalar delta, const scalar lambda, const int i);
		scalar bDot(const float t, const scalar delta, const scalar lambda, const int i);
		scalar bDDot(const float t, const scalar delta, const scalar lambda, const int i);

		THREADED_METHOD3(Wiggly, shouldMultithread(), u, VecX&, out, const float, t, const VecX&, coeffs);
		void uPartial(VecX& out, const float t, const VecX& coeffs, const UT_JobInfo& info);
		
		THREADED_METHOD3(Wiggly, shouldMultithread(), uDot, VecX&, out, const float, t, const VecX&, coeffs);
		void uDotPartial(VecX& out, const float t, const VecX& coeffs, const UT_JobInfo& info);

		// Wiggly Splines
		scalar wiggly(const float t, const int d, const scalar delta, const scalar lambda, const VecX& coeffs);
		scalar wigglyDot(const float t, const int d, const scalar delta, const scalar lambda, const VecX& coeffs);
		scalar wigglyDDot(const float t, const int d, const scalar delta, const scalar lambda, const VecX& coeffs);

		// Energy Functions
		THREADED_METHOD6(
			Wiggly, k.u.size()>1, perKeyEnergy, scalar&, total, const Keyframe&, k, 
			const VecX&, uPos, const VecX&, uVel, const GA_ROHandleV3D&, v_h, const GA_ROHandleI&, og_h);
		void perKeyEnergyPartial(
			scalar& total, const Keyframe& k, const VecX& uPos, const VecX& uVel, 
			const GA_ROHandleV3D& v_h, const GA_ROHandleI& og_h, const UT_JobInfo& info);
		scalar keyframeEnergy(const VecX& coeffs);

		THREADED_METHOD2(Wiggly, shouldMultithread(), dynamicsEnergy, scalar&, sum, const VecX&, coeffs);
		void dynamicsEnergyPartial(scalar& sum, const VecX& coeffs, const UT_JobInfo& info);
		scalar integralEnergy(const int d, const scalar delta, const scalar lambda, const VecX& coeffs);

		int getSegmentIdx(const float t);
		float getNormalizedTime(const float frame);

		/* Get the c constant */
		inline scalar getC(const int& d, const scalar& lambda) { return abs(lambda) > parms.eps ? parms.g[d%3] / abs(lambda) : 0.0; }
		/* Get the damping constant */
		inline scalar getDelta(const scalar& lambda) { return 0.5 * (parms.alpha + parms.beta * lambda); }
		/* Get the lambda constant */
		inline scalar getLambda(const int& i) { return eigenValues(i); }

		inline bool shouldMultithread() { return parms.d > 1; }

		int getNumCoeffs() { return 4 * (keyframes.size() - 1); }

		/* Get the total number of DOF (i.e. x, y, z for every node)*/
		int getDof() { return mDof; }
		/* Get the total number of coefficients across all wiggly splines */
		int getTotalNumCoeffs() { return 4 * (keyframes.size() - 1) * parms.d; }

		/* Get the flattened idx of a coefficient */
		inline int getCoeffIdx(const int k, const int d, const int l) { return k * 4 * parms.d + 4 * d + l; }


		int mDof;

		float startFrame;
		float endFrame;

		const GU_Detail* mesh;
		Keyframes keyframes;
		WigglyParms parms;

		MatX M;
		MatX K;

		VecX eigenValues;   // getDof() x 1
		MatX eigenModes;    // getDof() x getDof()
		VecX coefficients;  // getDof() x 1

		// Caching the end displacement and velocity for multiple splines
		VecX uEnd;
		VecX uDotEnd;

	};
}
