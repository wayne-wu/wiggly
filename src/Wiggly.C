#include "GU/GU_PrimTetrahedron.h"
#include "UT/UT_Interrupt.h"


#include "Wiggly.h"

#include "Eigen/Eigenvalues"
#include "Eigen/SVD"
#include "Eigen/QR"
#include "unsupported/Eigen/NonLinearOptimization"
#include "unsupported/Eigen/NumericalDiff"

#include "gsl/gsl_integration.h"

#include <iostream>
#include <unordered_set>


#define CANTOR(a, b) (a + b) * (a + b + 1) / 2 + a


using namespace HDK_Wiggly;

/*
Functor for minimizing the energy function using non-linear optimization.
*/
struct WigglyFunctor
{
	typedef float Scalar;

	typedef Eigen::VectorXf InputType;
	typedef Eigen::VectorXf ValueType;
	typedef Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> JacobianType;

	enum {
		InputsAtCompileTime = Eigen::Dynamic,
		ValuesAtCompileTime = Eigen::Dynamic
	};

	WigglyFunctor(Wiggly& obj, const Eigen::MatrixXf& U, const Eigen::VectorXf& w0) 
		: wiggly(obj), U(U), w0(w0) {}

	int operator()(const Eigen::VectorXf& z, Eigen::VectorXf& fvec) const
	{
		//NOTE: The algorithm will square fvec internally. Problematic?
		fvec(0) = wiggly.totalEnergy(w0 + U*z);
		return 0;
	}

	Wiggly& wiggly;
	const Eigen::MatrixXf& U;
	const Eigen::VectorXf& w0;

	int inputs() const { return wiggly.getNumPoints(); }// inputs is the dimension of x.
	int values() const { return 1; } // "values" is the number of f_i and
};

/*
Struct for storing data that can be passed to the gsl integration function
*/
struct gsl_fdata
{
	Wiggly* w;
	float lambda;
	float delta;
	const VecX* coeffs;
};

/*
gsl integration function
*/
double f(double t, void* params)
{
	gsl_fdata* data = (gsl_fdata*)params;
	return data->w->integrand(t, data->delta, data->lambda, *data->coeffs);
}

/*
Get the current keyframe index given the time
*/
int Wiggly::getKeyframeIdx(const float t)
{
	for (int i = 0; i < keyframes.size() - 1; ++i)
		if (keyframes[i].frame <= t && keyframes[i + 1].frame >= t)
			return i;
	return 0;
}

/*
Basis function
*/
float Wiggly::b(const float t, const float delta, const float lambda, int i)
{
	int sign1 = i <= 1 ? 1 : - 1;
	int sign2 = i % 2 == 0 ? 1 : -1;
	
	if (delta == 0 and lambda == 0)
		return pow(t, i);

	float tmp = delta * delta - lambda;
	if (tmp > 0)
		return exp(t * ((sign1 * delta) + (sign2 * sqrt(tmp))));
	else if (sign2 == 1)
		return exp(sign1 * t * delta) * cos(t * sqrt(-tmp));
	else
		return -exp(sign1 * t * delta) * sin(t * sqrt(-tmp));
}

/*
First derivative of the basis function
*/
float Wiggly::bDot(const float t, const float delta, const float lambda, int i)
{
	int sign1 = i <= 1 ? 1 : -1;
	int sign2 = i % 2 == 0 ? 1 : -1;

	if (delta == 0 and lambda == 0)
		return pow(t, i);

	float tmp = delta * delta - lambda;
	if (tmp > 0)
	{
		float expr = sign1 * delta + sign2 * sqrt(tmp);
		return exp(t * expr) * expr;
	}
	else
	{
		float expr = sqrt(-tmp);
		if (sign2 == 1)
			return exp(sign1 * t * delta) * ((sign1 * delta * cos(t * expr)) - (expr * sin(t * expr)));
		else
			return exp(sign1 * t * delta) * (-(expr * cos(t * expr)) - (sign1 * delta * sin(t * expr)));
	}
}

/*
Second derivative of the basis function
*/
float Wiggly::bDDot(const float t, const float delta, const float lambda, int i)
{
	int sign1 = i <= 1 ? 1 : -1;
	int sign2 = i % 2 == 0 ? 1 : -1;

	if (delta == 0 and lambda == 0)
		return pow(t, i);

	float tmp = delta * delta - lambda;
	if (tmp > 0)
	{
		float expr = sign1 * delta + sign2 * sqrt(tmp);
		return exp(t * expr) * expr * expr;
	}
	else
	{
		float expr = sqrt(-tmp);
		float dsqExpr = 2 * delta * delta - lambda;
		if (sign2 == 1)
			return exp(sign1 * t) * ((dsqExpr * cos(t * expr)) + (2 * delta * expr * (-sign1) * sin(t * expr)));
		else
			return -sign1 * exp(sign1 * t * delta) * ((2 * delta * expr * cos(t * expr)) + (sign1 * dsqExpr * sin(t * expr)));
	}
}


/*
Compute the wiggly spline value
*/
float Wiggly::wiggly(const float t, const float delta, const float lambda, const VecX& coeffs)
{
	int i = getKeyframeIdx(t);

	// Evaluate spline
	float c = lambda != 0 ? parms.g / abs(lambda) : 0;
	
	float sum = 0;
	sum += coeffs[4 * i + 0] * b(t, delta, lambda, 0);
	sum += coeffs[4 * i + 1] * b(t, delta, lambda, 1);
	sum += coeffs[4 * i + 2] * b(t, delta, lambda, 2);
	sum += coeffs[4 * i + 3] * b(t, delta, lambda, 3);

	return sum - c;
}	

/*
Compute the first derivative of the wiggly spline
*/
float Wiggly::wigglyDot(const float t, const float delta, const float lambda, const VecX& coeffs)
{
	int i = getKeyframeIdx(t);

	float sum = 0;
	sum += coeffs[4 * i + 0] * bDot(t, delta, lambda, 0);
	sum += coeffs[4 * i + 1] * bDot(t, delta, lambda, 1);
	sum += coeffs[4 * i + 2] * bDot(t, delta, lambda, 2);
	sum += coeffs[4 * i + 3] * bDot(t, delta, lambda, 3);

	return sum;
}

/*
Compute the first derivative of the wiggly spline
*/
float Wiggly::wigglyDDot(const float t, const float delta, const float lambda, const VecX& coeffs)
{
	int i = getKeyframeIdx(t);

	float sum = 0;
	sum += coeffs[4 * i + 0] * bDDot(t, delta, lambda, 0);
	sum += coeffs[4 * i + 1] * bDDot(t, delta, lambda, 1);
	sum += coeffs[4 * i + 2] * bDDot(t, delta, lambda, 2);
	sum += coeffs[4 * i + 3] * bDDot(t, delta, lambda, 3);

	return sum;
}

VecX Wiggly::u(const float t)
{
	VecX displacement = VecX::Zero(3 * getNumPoints());
	for (int i = 0; i < getNumPoints(); i++)
	{
		displacement[3 * i] = sin(t);
	}
	return displacement;
}

/*
Compute the displacement vector
*/
VecX Wiggly::u(const float t, const VecX& coeffs)
{
	VecX out = Eigen::VectorXf::Zero(3 * getNumPoints());
	for (int i = 0; i < parms.d; i++)
	{
		float lambda = eigenValues(i);
		float delta = 0.5 * (parms.alpha + parms.beta * lambda);
		out += wiggly(t, delta, lambda, coeffs) * eigenModes.col(i);
	}
	return out;
}

/*
Compute the velocity vector
*/
VecX Wiggly::uDot(const float t, const VecX& coeffs)
{
	VecX out = Eigen::VectorXf::Zero(3 * getNumPoints());
	for (int i = 0; i < parms.d; i++)
	{
		float lambda = eigenValues(i);
		float delta = 0.5 * (parms.alpha + parms.beta * lambda);
		out += wigglyDot(t, delta, lambda, coeffs) * eigenModes.col(i);
	}
	return out;
}

/*
Compute the energy based on constraints
*/
float Wiggly::keyframeEnergy(const VecX& coeffs)
{
	float total = 0;
	for (int i = 0; i < keyframes.size(); ++i)
	{
		float frame = keyframes[i].frame;
		Eigen::VectorXf uPos = u(frame, coeffs);
		Eigen::VectorXf uVel = uDot(frame, coeffs);

		float posDiff = 0, velDiff = 0;
		for (int j = 0; j < keyframes[i].points.size(); ++j)
		{
			int ptIdx = keyframes[i].points[j];
			if (ptIdx < 0) 
				continue;

			if (keyframes[i].hasPos) 
			{
				UT_Vector3 ak = getPosConstraint(keyframes[i].detail, ptIdx);
				UT_Vector3 uk = UT_Vector3(uPos(3 * ptIdx), uPos(3 * ptIdx + 1), uPos(3 * ptIdx + 2));
				posDiff += (ak - uk).length2();
			}

			if (keyframes[i].hasVel)
			{
				UT_Vector3 ak = getVelConstraint(keyframes[i].detail, ptIdx);
				UT_Vector3 uk = UT_Vector3(uVel(3 * ptIdx), uVel(3 * ptIdx + 1), uVel(3 * ptIdx + 2));
				velDiff += (ak - uk).length2();
			}
		}
		total += posDiff * parms.cA;
		total += velDiff * parms.cB;
	}
	return 0.5 * total;
}

/*
The integrand for the energy minimizing integrand
*/
float Wiggly::integrand(const float t, const float delta, const float lambda, const VecX& coeffs)
{
	float tmp = wigglyDDot(t, delta, lambda, coeffs) + 
		2 * delta * wigglyDot(t, delta, lambda, coeffs) + 
		lambda * wiggly(t, delta, lambda, coeffs) + parms.g;
	return 0.5 * tmp * tmp;
}


/*
Evaluate the energy of a wiggly spline based on the integral
*/
float Wiggly::integralEnergy(const float lambda, const float delta, const VecX& coeffs)
{
	gsl_integration_workspace* w = gsl_integration_workspace_alloc(1000);

	double result, error;
	double expected = -4.0;
	double alpha = 1.0;

	gsl_function F;
	F.function = &f;

	gsl_fdata data;
	data.w = this;
	data.delta = delta;
	data.lambda = lambda;
	data.coeffs = &coeffs;

	F.params = &data;

	float a = keyframes[0].frame;
	float b = keyframes[keyframes.size() - 1].frame;

	gsl_integration_qags(&F, a, b, 0, 1e-7, 1000, w, &result, &error);

	return float(result);
}

/*
Evaluate the energy of the linear dynamics decoupled using wiggly splines
*/
float Wiggly::dynamicsEnergy(const VecX& coeffs)
{
	float total = 0;
	int numCoeff = getNumCoeffs();
	for (int i = 0; i < parms.d; ++i)
	{
		float lambda = eigenValues(i);
		float delta = 0.5f * (parms.alpha + parms.beta * lambda);
		float e = integralEnergy(delta, lambda, coeffs.segment(numCoeff * i, numCoeff));
		total += e * e;
	}
	return 0.5*total;
}

/*
Helper to get the position constraint for a given point index.
*/
UT_Vector3 Wiggly::getPosConstraint(const GU_Detail* detail, const GA_Index ptidx)
{
	return detail->getPos3(detail->pointOffset(ptidx));
}

/*
Helper to get the velocity constraint for a given velocity index.
*/
UT_Vector3 Wiggly::getVelConstraint(const GU_Detail* detail, const GA_Index ptidx)
{
	GA_ROHandleV3 v_h(detail, GA_ATTRIB_POINT, "v");
	return v_h.get(detail->pointOffset(ptidx));
}

/*
Compute the wiggly splines' coefficients based on the sparse keyframes.
This should be recomputed when the mesh changes or the keyframes change.
*/
void Wiggly::compute() {

	UT_AutoInterrupt progress("Computing wiggly spline coefficients");

	int dof = getDof();
	int d = parms.d;

	int numCoeffs = getTotalNumCoeffs();
	int numConditions = 4 * d;
	for (int i = 1; i < keyframes.size() - 1; i++)
		numConditions += keyframes[i].hasVel ? 2 * d : 3 * d;

	Eigen::MatrixXf A = Eigen::MatrixXf::Zero(numConditions, numCoeffs);
	Eigen::VectorXf B = Eigen::VectorXf::Zero(numConditions);

	int row = 0; // The row correlates to the number of linear conditions

	// Boundary condition at t0
	const Keyframe& k0 = keyframes.front();
	for (int i = 0; i < d; i++)
	{

		float lambda = eigenValues(i);
		float delta = 0.5f * (parms.alpha + parms.beta * lambda);

		std::cout << "Dim: " << i << " Lambda: " << lambda << " Delta: " << delta << std::endl;
		for (int j = 0; j < 4; j++)
		{
			int col = getCoeffIdx(0, i, j);
			float tmp = b(k0.frame, delta, lambda, j);
			std::cout << " Basis " << j << " " << tmp;
			A(row + i, col) = tmp;
			if (k0.hasVel)
				A(row + d + i, col) = bDot(k0.frame, delta, lambda, j);
		}
		std::cout << std::endl;
	}

	UT_Array<UT_Vector3F> positions;
	k0.detail->getPos3AsArray(k0.detail->getPointRange(), positions);
	Eigen::Map<Eigen::VectorXf> p0(positions.data()->data(), dof);

	Eigen::MatrixXf phi = eigenModes(Eigen::all, Eigen::seqN(0,d)).transpose();  // This should be d x 3n

	B.segment(row, d) = phi * M * p0;
	//if (k0.hasVel)
	//	B.segment(row + d, d) = Eigen::VectorXf::Zero(d);

	row += k0.hasVel ? 2*d : 1*d;

	// In-between constraints
	for (int i = 1; i < keyframes.size() - 1; i++)
	{
		const Keyframe& ki = keyframes[i];
		for (int j = 0; j < d; j++)
		{
			float lambda = eigenValues(i);
			float delta = 0.5f * (parms.alpha + parms.beta * lambda);

			for (int l = 0; l < 4; l++)
			{
				float bj = b(ki.frame, delta, lambda, j);
				float bdj = bDot(ki.frame, delta, lambda, j);
				float bddj = bDDot(ki.frame, delta, lambda, j);

				int col1 = getCoeffIdx(i - 1, j, l);
				int col2 = getCoeffIdx(i, j, l);

				// C0 Continuity
				A(row + j, col1) = bj;
				A(row + j, col2) = -bj;

				// C1 Continuity
				A(row + d + j, col1) = bdj;
				A(row + d + j, col2) = -bdj;

				// C2 Continuity
				A(row + 2*d + j, col1) = bddj;
				A(row + 2*d + j, col2) = -bddj;

				// TODO: Handle velocity case
			}
		}
	}

	row += 3*d;

	// Boundary condition at t1
	const Keyframe& km = keyframes.back();
	for (int i = 0; i < d; i++)
	{
		float lambda = eigenValues(i);
		float delta = 0.5f * (parms.alpha + parms.beta * lambda);
		for (int j = 0; j < 4; j++)
		{
			int col = getCoeffIdx(keyframes.size() - 1, i, j);
			A(row + i, col) = b(km.frame, delta, lambda, j);
			if (km.hasVel)
				A(row + d + i, col) = bDot(km.frame, delta, lambda, j);
		}
	}

	km.detail->getPos3AsArray(km.detail->getPointRange(), positions);
	Eigen::Map<Eigen::VectorXf> pm(positions.data()->data(), dof);

	B.segment(row, d) = phi * M * pm;
	//if (km.hasVel)
	//	B.segment(row+d, d) = Eigen::VectorXf::Zero(d);

	if (progress.wasInterrupted())
		return;

	// NOTE: BDCSVD preferred for larger matrix
	Eigen::BDCSVD<Eigen::MatrixXf> svd(A, Eigen::ComputeThinU | Eigen::ComputeThinV);
	int subspace_dim = numCoeffs - numConditions;
	
	std::cout << svd.info() << std::endl;

	// Eigen::MatrixXf U = svd.matrixV()(Eigen::all, Eigen::seq(numCoeffs - subspace_dim, Eigen::last));

	// Eigen::VectorXf w0 = svd.solve(B);

	// std::cout << w0 << std::endl;

	//if (progress.wasInterrupted())
	//	return;

	//// NOTE: Is passing in *this the best way to approach functor?
	//WigglyFunctor functor(*this, U, w0);
	//Eigen::NumericalDiff<WigglyFunctor> numDiff(functor);
	//Eigen::LevenbergMarquardt<Eigen::NumericalDiff<WigglyFunctor>, float> lm(numDiff);

	//// Initial guess
	//Eigen::VectorXf xmin = Eigen::VectorXf::Zero(numCoeffs);

	//lm.minimize(xmin);

	//coefficients = xmin;  // TODO: Can we directly minimize on the member variable?
}

/*
Compute eigenvalues and eigenvectors based on stiffness and mass matrices
as part of the precompute step. This only needs to be recomputed if the mesh changes.
*/
void Wiggly::preCompute() {

	UT_AutoInterrupt progress1("Assembling stiffness and mass matrices.");

	int dof = getDof();

	/*
	Compute the global stiffness matrix of the tetrahedral mesh.
	Adopted from http://knapiontek.github.io/fem.pdf
	*/
	K = Eigen::MatrixXf::Zero(dof, dof);

	/*
	Compute the global mass matrix of the tetrahedral mesh.
	Adopted from 12.2.8 in Finite Element Method in Engineering (6th Edition)
	*/
	M = Eigen::MatrixXf::Zero(dof, dof);

	const float EA = 1000;

	std::unordered_set<int> visitedEdges;

	for (GA_Iterator it(mesh->getPrimitiveRange()); !it.atEnd(); ++it)
	{
		if (progress1.wasInterrupted())
			return;

		const GU_PrimTetrahedron* tet = (const GU_PrimTetrahedron*)mesh->getPrimitive(*it);

		// TODO: What is the refpt for?
		fpreal vol = parms.p * tet->calcVolume(UT_Vector3(0, 0, 0)) / 20.0;

		for (int i = 0; i < 6; i++)
		{
			int i0; int i1;
			tet->getEdgeIndices(i, i0, i1);

			const GA_Index pt1idx = tet->getPointIndex(i0);
			const GA_Index pt2idx = tet->getPointIndex(i1);

			if (!visitedEdges.insert(
				pt1idx < pt2idx ? CANTOR(pt1idx, pt2idx) : CANTOR(pt2idx, pt1idx)).second)
				continue;

			const UT_Vector3& point1 = tet->getPos3(i0);
			const UT_Vector3& point2 = tet->getPos3(i1);

			int p1x = 3 * pt1idx + 0;
			int p1y = 3 * pt1idx + 1;
			int p1z = 3 * pt1idx + 2;
			int p2x = 3 * pt2idx + 0;
			int p2y = 3 * pt2idx + 1;
			int p2z = 3 * pt2idx + 2;

			float dx = point2.x() - point1.x();
			float dy = point2.y() - point1.y();
			float dz = point2.z() - point1.z();
			float l = 1.0 / sqrt(dx * dx + dy * dy + dz * dz);

			float cx = dx * l;
			float cy = dy * l;
			float cz = dz * l;

			float cxx = cx * cx * EA * l;
			float cyy = cy * cy * EA * l;
			float czz = cz * cz * EA * l;
			float cxy = cx * cy * EA * l;
			float cxz = cx * cz * EA * l;
			float cyz = cy * cz * EA * l;

			K(p1x, p1x) += cxx;
			K(p1y, p1x) += cxy;
			K(p1z, p1x) += cxz;
			K(p2x, p1x) -= cxx;
			K(p2y, p1x) -= cxy;
			K(p2z, p1x) -= cxz;

			K(p1x, p1y) += cxy;
			K(p1y, p1y) += cyy;
			K(p1z, p1y) += cyz;
			K(p2x, p1y) -= cxy;
			K(p2y, p1y) -= cyy;
			K(p2z, p1y) -= cyz;

			K(p1x, p1z) += cxz;
			K(p1y, p1z) += cyz;
			K(p1z, p1z) += czz;
			K(p2x, p1z) -= cxz;
			K(p2y, p1z) -= cyz;
			K(p2z, p1z) -= czz;

			K(p1x, p2x) -= cxx;
			K(p1y, p2x) -= cxy;
			K(p1z, p2x) -= cxz;
			K(p2x, p2x) += cxx;
			K(p2y, p2x) += cxy;
			K(p2z, p2x) += cxz;

			K(p1x, p2y) -= cxy;
			K(p1y, p2y) -= cyy;
			K(p1z, p2y) -= cyz;
			K(p2x, p2y) += cxy;
			K(p2y, p2y) += cyy;
			K(p2z, p2y) += cyz;

			K(p1x, p2z) -= cxz;
			K(p1y, p2z) -= cyz;
			K(p1z, p2z) -= czz;
			K(p2x, p2z) += cxz;
			K(p2y, p2z) += cyz;
			K(p2z, p2z) += czz;

			M(p1x, p1x) += 2 * vol;
			M(p1y, p1y) += 2 * vol;
			M(p1z, p1z) += 2 * vol;

			M(p1x, p2x) += 1 * vol;
			M(p1y, p2y) += 1 * vol;
			M(p1z, p2z) += 1 * vol;

			M(p2x, p1x) += 1 * vol;
			M(p2y, p1y) += 1 * vol;
			M(p2z, p1z) += 1 * vol;
			
			M(p2x, p2x) += 2 * vol;
			M(p2y, p2y) += 2 * vol;
			M(p2z, p2z) += 2 * vol;
		}
	}

	// std::cout << K << std::endl;
	// std::cout << M << std::endl;

	// printf("Determinant: %f\n", K.determinant());

	UT_AutoInterrupt progress2("Finding eigenvalues and eigen vectors.");
	if (progress2.wasInterrupted())
		return;

	// Find eigenvalues and eigenvectors
	Eigen::GeneralizedEigenSolver<Eigen::MatrixXf> ges;
	ges.compute(K, M);

	eigenValues = ges.eigenvalues().real();
	eigenModes = ges.eigenvectors().real();
}