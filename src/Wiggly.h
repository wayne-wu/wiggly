#pragma once

#include <vector>
#include <utility>
#include <unordered_map>

#include <UT/UT_Vector3.h>
#include <GU/GU_Detail.h>

#include "Eigen/Dense"


// NOTE: Not sure if we should include Houdini dependencies here.
// On one hand, it would save memory to not have to duplicate data,
// but then you can't decouple the system anymore


namespace HDK_Wiggly {

	typedef double scalar;  // <------- Set precision here
	typedef Eigen::VectorXd VecX;
	typedef Eigen::MatrixXd MatX;

	typedef std::pair<int, int> Element;

	struct Keyframe
	{
		float t;
		float frame;
		std::vector<GA_Index> points;
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
		scalar g = 1;
		int d = 20;
		scalar cA = 1.0;
		scalar cB = 1.0;
		scalar p = 1000;
		scalar young;
		scalar eps = 1e-6;
		scalar poisson;
	};

	class Wiggly
	{

	public:
		Wiggly(const GU_Detail* mgdp, const WigglyParms& sopparms)
			: mesh(mgdp), parms(sopparms) {}
		~Wiggly() {}

		void compute();
		void preCompute();

		VecX u(const float t);
		scalar totalEnergy(const VecX& c);
		scalar integrand(const float t, const int d, const scalar delta, const scalar lambda, const VecX& coeffs);
		int getNumPoints() { return mesh->getNumPoints(); }
		Keyframes& getKeyframes() { return keyframes; }

	protected:

		// Basis Functions
		scalar b(const float t, const scalar delta, const scalar lambda, const int i);
		scalar bDot(const float t, const scalar delta, const scalar lambda, const int i);
		scalar bDDot(const float t, const scalar delta, const scalar lambda, const int i);

		// Displacement Functions
		VecX u(const float t, const VecX& coeffs);
		VecX uDot(const float t, const VecX& coeffs);

		// Wiggly Splines
		scalar wiggly(const float t, const int d, const scalar delta, const scalar lambda, const VecX& coeffs);
		scalar wigglyDot(const float t, const int d, const scalar delta, const scalar lambda, const VecX& coeffs);
		scalar wigglyDDot(const float t, const int d, const scalar delta, const scalar lambda, const VecX& coeffs);

		// Energy Functions
		scalar keyframeEnergy(const VecX& coeffs);
		scalar dynamicsEnergy(const VecX& coeffs);
		scalar integralEnergy(const int d, const scalar delta, const scalar lambda, const VecX& coeffs);

		int getSegmentIdx(const float t);
		float getNormalizedTime(const float frame);

		/* Get the c constant */
		inline scalar getC(const scalar& lambda) { return abs(lambda) > parms.eps ? parms.g / abs(lambda) : 0.0; }
		/* Get the damping constant */
		inline scalar getDelta(const scalar& lambda) { return 0.5 * (parms.alpha + parms.beta * lambda); }
		/* Get the lambda constant */
		inline scalar getLambda(const int& i) { return eigenValues(i); }


		int getNumCoeffs() { return 4 * (keyframes.size() - 1); }
		/* Get the total number of DOF (i.e. x, y, z for every node)*/
		int getDof() { return 3 * getNumPoints(); }
		/* Get the total number of coefficients across all wiggly splines */
		int getTotalNumCoeffs() { return 4 * (keyframes.size() - 1) * parms.d; }

		/* Get the flattened idx of a coefficient */
		inline int getCoeffIdx(const int k, const int d, const int l) { return k * 4 * parms.d + 4 * d + l; }

		const GU_Detail* mesh;
		Keyframes keyframes;
		const WigglyParms parms;

		MatX M;
		MatX K;

		VecX eigenValues;   // getDof() x 1
		MatX eigenModes;    // getDof() x getDof()
		VecX coefficients;  // getDof() x 1
	};
}
