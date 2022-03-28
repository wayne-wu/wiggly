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

	typedef std::pair<int, int> Element;
	typedef Eigen::VectorXf VecX;

	struct Keyframe
	{
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
		float g = 9.81;
		int dim = 20;
		float cA = 1.0;
		float cB = 1.0;
	};

	class Wiggly
	{

	public:
		Wiggly(const GU_Detail* mgdp, const Keyframes& keyframes, const WigglyParms& parms) 
			: mesh(mgdp), keyframes(keyframes), parms(parms) {}
		~Wiggly() {}

		void compute();
		VecX u(const float t);
		float totalEnergy(const VecX& c) { return dynamicsEnergy(c) + keyframeEnergy(c); }
		float integrand(const float t, const float delta, const float lambda, const VecX& coeffs);
		int getNumPoints() { return mesh->getNumPoints(); }
		
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
		float wiggly(const float t, const float delta, const float lambda, const VecX& coeffs);
		float wigglyDot(const float t, const float delta, const float lambda, const VecX& coeffs);
		float wigglyDDot(const float t, const float delta, const float lambda, const VecX& coeffs);

		// Energy Functions
		float keyframeEnergy(const VecX& coeffs);
		float dynamicsEnergy(const VecX& coeffs);
		float integralEnergy(const float delta, const float lambda, const VecX& coeffs);

		void computeStiffnessMatrix();
		void computeMassMatrix();
		void computeCoefficients();

		int getKeyframeIdx(const float t);
		int getNumCoeffs() { return 4 * (keyframes.size() - 1); }

		UT_Vector3 getPosConstraint(const GU_Detail* detail, const GA_Index ptIdx);
		UT_Vector3 getVelConstraint(const GU_Detail* detail, const GA_Index ptIdx);

		const GU_Detail* mesh;
		const Keyframes& keyframes;
		const WigglyParms& parms;

		Eigen::MatrixXf M;
		Eigen::MatrixXf K;

		Eigen::VectorXf eigenValues;
		Eigen::MatrixXf eigenModes;
		Eigen::VectorXf coefficients;  //4*m*d
	};
}
