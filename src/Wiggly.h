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

#define EPS 1e-10


namespace HDK_Wiggly {

	typedef std::pair<int, int> Element;
	typedef Eigen::VectorXf VecX;

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
		float alpha;
		float beta;
		float g = 1;
		int d = 20;
		float cA = 1.0;
		float cB = 1.0;
		float p = 1000;
		float young;
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
		float totalEnergy(const VecX& c);
		float integrand(const float t, const int d, const float delta, const float lambda, const VecX& coeffs);
		int getNumPoints() { return mesh->getNumPoints(); }
		Keyframes& getKeyframes() { return keyframes; }

	protected:

		// Basis Functions
		float b(const float t, const float delta, const float lambda, const int i);
		float bDot(const float t, const float delta, const float lambda, const int i);
		float bDDot(const float t, const float delta, const float lambda, const int i);

		// Displacement Functions
		VecX u(const float t, const VecX& coeffs);
		VecX uDot(const float t, const VecX& coeffs);
		//VecX& uDDot(const float t, const VecX& coeffs);

		// Wiggly Splines
		float wiggly(const float t, const int d, const float delta, const float lambda, const VecX& coeffs);
		float wigglyDot(const float t, const int d, const float delta, const float lambda, const VecX& coeffs);
		float wigglyDDot(const float t, const int d, const float delta, const float lambda, const VecX& coeffs);

		// Energy Functions
		float keyframeEnergy(const VecX& coeffs);
		float dynamicsEnergy(const VecX& coeffs);
		float integralEnergy(const int d, const float delta, const float lambda, const VecX& coeffs);

		int getSegmentIdx(const float t);
		float getNormalizedTime(const float frame);

		/* Get the c constant */
		inline float getC(const float& lambda) { return abs(lambda) > EPS ? parms.g / abs(lambda) : 0.0; }
		/* Get the damping constant */
		inline float getDelta(const float& lambda) { return 0.5f * (parms.alpha + parms.beta * lambda); }
		/* Get the lambda constant */
		inline float getLambda(const int& i) { return eigenValues(i); }
		

		int getNumCoeffs() { return 4 * (keyframes.size() - 1); }
		/* Get the total number of DOF (i.e. x, y, z for every node)*/
		int getDof() { return 3 * getNumPoints(); }
		/* Get the total number of coefficients across all wiggly splines */
		int getTotalNumCoeffs() { return 4 * (keyframes.size() - 1) * parms.d; }
		
		/* Get the flattened idx of a coefficient */
		inline int getCoeffIdx(const float k, const float d, const float l) { return k * 4 * parms.d + 4 * d + l; }

		const GU_Detail* mesh;
		Keyframes keyframes;
		const WigglyParms parms;

		Eigen::MatrixXf M;
		Eigen::MatrixXf K;

		Eigen::VectorXf eigenValues;   // getDof() x 1
		Eigen::MatrixXf eigenModes;    // getDof() x getDof()
		Eigen::VectorXf coefficients;  // getDof() x 1
	};
}
