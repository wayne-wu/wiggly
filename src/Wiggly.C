#include "GU/GU_PrimTetrahedron.h"
#include "UT/UT_Interrupt.h"

#include "Wiggly.h"

#include "Eigen/Eigenvalues"
#include "Eigen/SVD"
#include "Eigen/QR"
#include "unsupported/Eigen/NonLinearOptimization"
#include "unsupported/Eigen/NumericalDiff"

#include "gsl/gsl_integration.h"
#include "gsl/gsl_multimin.h"

#include <dlib/optimization.h>

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

	WigglyFunctor(Wiggly& obj, const Eigen::MatrixXf& U, const Eigen::VectorXf& w0, const int& dim) 
		: wiggly(obj), U(U), w0(w0), subspaceDim(subspaceDim) {}

	int operator()(const Eigen::VectorXf& z, Eigen::VectorXf& fvec) const
	{
		//NOTE: The algorithm will square fvec internally. Problematic?
		fvec(0) = wiggly.totalEnergy(w0 + U*z);
		std::cout << "energy: " << z << std::endl;
		return 0;
	}

	Wiggly& wiggly;
	const Eigen::MatrixXf& U;
	const Eigen::VectorXf& w0;
	const int subspaceDim;


	int inputs() const { return subspaceDim; } // inputs is the dimension of x.
	int values() const { return 1; } // "values" is the number of f_i and
};


typedef dlib::matrix<double, 0, 1> column_vector;

struct ObjFunctor
{
	ObjFunctor(Wiggly& obj, const Eigen::MatrixXf& U, const Eigen::VectorXf& w0, const int& dim)
		: wiggly(obj), U(U), w0(w0), subDim(dim) {}

	double operator()(const column_vector& v) const
	{
		Eigen::VectorXf z = Eigen::VectorXf::Zero(subDim);
		for (int i = 0; i < subDim; i++)
			z(i) = v(i);

		float e = wiggly.totalEnergy(w0 + U * z);

		return e;
	}

	Wiggly& wiggly;
	const Eigen::MatrixXf& U;
	const Eigen::VectorXf& w0;
	const int subDim;
};

/*
Struct for storing data that can be passed to the gsl integration function
*/
struct gsl_integrand_data
{
	Wiggly* w;
	float lambda;
	float delta;
	const VecX* coeffs;
};

struct gsl_objective_data
{
	Wiggly* w;
	Eigen::MatrixXf U;
	Eigen::VectorXf w0;
	int subspaceDim;
};

/*
gsl integration function
*/
double gsl_integrand(double t, void* params)
{
	// gsl_integrand_data* data = (gsl_integrand_data*)params;
	
	// return data->w->integrand(t, data->delta, data->lambda, *data->coeffs);
	return sin(t);
}

//double gsl_objective(const gsl_vector, )

/*
Get the current keyframe index given the time
*/
int Wiggly::getSegmentIdx(const float t)
{
	for (int i = 0; i < keyframes.size() - 1; ++i)
		if (keyframes[i].t <= t && keyframes[i + 1].t >= t)
			return i;
	return 0;
}

/*
Get the normalized time (i.e 0 - 1) give the frame number
*/
float Wiggly::getNormalizedTime(const float frame)
{
	return (frame - keyframes.front().frame) / (keyframes.back().frame - keyframes.front().frame);
}


/*
Basis function
*/
float Wiggly::b(const float t, const float delta, const float lambda, int i)
{
	int sign1 = i <= 1 ? -1 : 1;
	int sign2 = i % 2 == 0 ? 1 : -1;
	
	// delta == 0 && lambda == 0
	if (abs(delta) < EPS && abs(lambda) < EPS)
		return pow(t, i);

	// delta != 0 && lambda == 0
	if (abs(delta) > EPS && abs(lambda) < EPS)
	{
		switch(i){
		case 0:
			return 1;
		case 1:
			return t;
		case 2:
			return exp(-2 * delta * t) / (4 * delta * delta);
		case 3:
			return exp(2 * delta * t) / (4 * delta * delta);
		default:
			return pow(t, i);
		}
	}

	float tmp = delta * delta - lambda;
	if (tmp > 0)
		return exp(t * (sign1 * delta + sign2 * sqrt(tmp)));
	else if (sign2 == 1) // b1 & b3
		return exp(sign1 * t * delta) * cos(t * sqrt(-tmp));
	else // b2 & b4
		return exp(sign1 * t * delta) * sin(t * sqrt(-tmp));
}

/*
First derivative of the basis function
*/
float Wiggly::bDot(const float t, const float delta, const float lambda, int i)
{
	int sign1 = i <= 1 ? -1 : 1;
	int sign2 = i % 2 == 0 ? 1 : -1;

	if (abs(delta) < EPS && abs(lambda) < EPS)
		return i == 0 ? 0 : i * pow(t, i - 1);

	if (abs(delta) > EPS && abs(lambda) < EPS)
	{
		switch (i) {
		case 0:
			return 0;
		case 1:
			return 1;
		case 2:
			return exp(-2 * delta * t) / (-2 * delta);
		case 3:
			return exp(2 * delta * t) / (2 * delta);
		default:
			return pow(t, i);
		}
	}

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
			return -exp(sign1 * t * delta) * (-(expr * cos(t * expr)) - (sign1 * delta * sin(t * expr)));
	}
}

/*
Second derivative of the basis function
*/
float Wiggly::bDDot(const float t, const float delta, const float lambda, int i)
{
	int sign1 = i <= 1 ? -1 : 1;
	int sign2 = i % 2 == 0 ? 1 : -1;

	if (abs(delta) < EPS && abs(lambda) < EPS)
		return i <= 1 ? 0 : i * (i - 1) * pow(t, i - 2);

	if (abs(delta) > EPS && abs(lambda) < EPS)
	{
		switch (i) {
		case 0:
			return 0;
		case 1:
			return 0;
		case 2:
			return exp(-2 * delta * t);
		case 3:
			return exp(2 * delta * t);
		default:
			return pow(t, i);
		}
	}

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
			return sign1 * exp(sign1 * t * delta) * ((2 * delta * expr * cos(t * expr)) + (sign1 * dsqExpr * sin(t * expr)));
	}
}

/*
Compute the wiggly spline value
*/
float Wiggly::wiggly(const float t, const int d, const float delta, const float lambda, const VecX& coeffs)
{
	int k = getSegmentIdx(t);

	// Evaluate spline
	float sum = 0;
	
	sum += coeffs[getCoeffIdx(k, d, 0)] * b(t, delta, lambda, 0);
	sum += coeffs[getCoeffIdx(k, d, 1)] * b(t, delta, lambda, 1);
	sum += coeffs[getCoeffIdx(k, d, 2)] * b(t, delta, lambda, 2);
	sum += coeffs[getCoeffIdx(k, d, 3)] * b(t, delta, lambda, 3);

	return sum - getC(lambda);
}	

/*
Compute the first derivative of the wiggly spline
*/
float Wiggly::wigglyDot(const float t, const int d, const float delta, const float lambda, const VecX& coeffs)
{
	int k = getSegmentIdx(t);

	float sum = 0;
	sum += coeffs[getCoeffIdx(k, d, 0)] * bDot(t, delta, lambda, 0);
	sum += coeffs[getCoeffIdx(k, d, 1)] * bDot(t, delta, lambda, 1);
	sum += coeffs[getCoeffIdx(k, d, 2)] * bDot(t, delta, lambda, 2);
	sum += coeffs[getCoeffIdx(k, d, 3)] * bDot(t, delta, lambda, 3);

	return sum;
}

/*
Compute the first derivative of the wiggly spline
*/
float Wiggly::wigglyDDot(const float t, const int d, const float delta, const float lambda, const VecX& coeffs)
{
	int k = getSegmentIdx(t);

	float sum = 0;
	sum += coeffs[getCoeffIdx(k, d, 0)] * bDDot(t, delta, lambda, 0);
	sum += coeffs[getCoeffIdx(k, d, 1)] * bDDot(t, delta, lambda, 1);
	sum += coeffs[getCoeffIdx(k, d, 2)] * bDDot(t, delta, lambda, 2);
	sum += coeffs[getCoeffIdx(k, d, 3)] * bDDot(t, delta, lambda, 3);

	return sum;
}

VecX Wiggly::u(const float f)
{

	float t = getNormalizedTime(f);
	return u(t, coefficients);
	 
	//VecX displacement = VecX::Zero(3 * getNumPoints());
	//for (int i = 0; i < getNumPoints(); i++)
	//{
	//	displacement[3 * i] = sin(t);
	//}
	//return displacement;
}

/*
Compute the displacement vector
*/
VecX Wiggly::u(const float t, const VecX& coeffs)
{
	VecX out = Eigen::VectorXf::Zero(getDof());
	for (int i = 0; i < parms.d; i++)
	{
		float lambda = getLambda(i);
		float delta = getDelta(lambda);
		out += wiggly(t, i, delta, lambda, coeffs) * eigenModes.col(i);
	}
	return out;
}

/*
Compute the velocity vector
*/
VecX Wiggly::uDot(const float t, const VecX& coeffs)
{
	VecX out = Eigen::VectorXf::Zero(getDof());
	for (int i = 0; i < parms.d; i++)
	{
		float lambda = getLambda(i);
		float delta = getDelta(lambda);
		out += wigglyDot(t, i, delta, lambda, coeffs) * eigenModes.col(i);
	}
	return out;
}

float Wiggly::totalEnergy(const VecX& c)
{
	float d = dynamicsEnergy(c);
	float k = keyframeEnergy(c);
	// std::cout << "d: " << d << "   k: " << k << std::endl;
	float e = d + k;
	return e;
}

/*
Compute the energy based on constraints
*/
float Wiggly::keyframeEnergy(const VecX& coeffs)
{
	float total = 0;
	for (Keyframe& k : keyframes)
	{
		Eigen::VectorXf uPos = u(k.t, coeffs);
		Eigen::VectorXf uVel = uDot(k.t, coeffs);

		GA_ROHandleV3 u_h(k.detail, GA_ATTRIB_POINT, "u");
		GA_ROHandleV3 v_h(k.detail, GA_ATTRIB_POINT, "v");

		float posDiff = 0, velDiff = 0;
		for (int j = 0; j < k.points.size(); ++j)
		{
			int ptIdx = k.points[j];
			if (ptIdx < 0) 
				continue;

			if (k.hasPos) 
			{
				UT_Vector3 ak = u_h.get(k.detail->pointOffset(ptIdx));
				UT_Vector3 uk = UT_Vector3(uPos(3 * ptIdx), uPos(3 * ptIdx + 1), uPos(3 * ptIdx + 2));
				posDiff += ak.distance2(uk);
			}

			if (k.hasVel)
			{
				UT_Vector3 bk = v_h.get(k.detail->pointOffset(ptIdx));
				UT_Vector3 vk = UT_Vector3(uVel(3 * ptIdx), uVel(3 * ptIdx + 1), uVel(3 * ptIdx + 2));
				velDiff += bk.distance2(vk);
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
float Wiggly::integrand(const float t, const int d, const float delta, const float lambda, const VecX& coeffs)
{
	float tmp = wigglyDDot(t, d, delta, lambda, coeffs) + 
		2 * delta * wigglyDot(t, d, delta, lambda, coeffs) + 
		lambda * wiggly(t, d, delta, lambda, coeffs) + parms.g;
	return tmp * tmp;
}

/*
Evaluate the energy of a wiggly spline based on the integral
*/
float Wiggly::integralEnergy(const int d, const float delta, const float lambda, const VecX& coeffs)
{	
	int intervals = 100;

	float a = keyframes.front().t;  // this should be 0 after normalizing
	float b = keyframes.back().t;  // this should be 1 after normalizing

	float stepSize = (b - a) / float(intervals);

	float t = a;
	float sum = integrand(a, d, delta, lambda, coeffs) + integrand(b, d, delta, lambda, coeffs);
	
	for (int i = 1; i <= intervals-1; ++i)
	{
		sum += 2.0 * integrand(a + i*stepSize, d, delta, lambda, coeffs);
	}

	sum = sum * stepSize * 0.25;
	
	return sum;
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
		float lambda = getLambda(i);
		float delta = getDelta(lambda);
		float e = integralEnergy(i, delta, lambda, coeffs);
		total += e;
	}

	return total;
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
	Eigen::VectorXf c = Eigen::VectorXf::Zero(d);
	Eigen::MatrixXf phiT = eigenModes(Eigen::all, Eigen::seqN(0, d)).transpose();  // This should be d x 3n

	std::cout << "------eigen check--------" << std::endl << phiT * M * phiT.transpose() << std::endl;
	std::cout << "------eigen check--------" << std::endl << phiT.transpose() * phiT * M << std::endl;

	int row = 0; // The row correlates to the number of linear conditions

	// Boundary condition at t0
	Keyframe& k0 = keyframes.front();
	k0.t = getNormalizedTime(k0.frame);  // TODO: Where's the best place to do this?
	for (int i = 0; i < d; i++)
	{
		float lambda = getLambda(i);
		float delta = getDelta(lambda);
		c(i) = getC(lambda);

		for (int j = 0; j < 4; j++)
		{
			int col = getCoeffIdx(0, i, j);
			A(row + i, col) = b(k0.t, delta, lambda, j);
			if (k0.hasVel)
				A(row + d + i, col) = bDot(k0.t, delta, lambda, j);
		}
	}

	UT_Array<UT_Vector3F> displacements;
	const GA_Attribute* u_attrib = k0.detail->findFloatTuple(GA_ATTRIB_POINT, "u", 3);
	// TODO: Might need to account for incorrect point order?
	k0.detail->getAttributeAsArray<UT_Vector3F>(u_attrib, k0.detail->getPointRange(), displacements);
	Eigen::Map<Eigen::VectorXf> u0(displacements.data()->data(), dof);

	B.segment(row, d) = phiT * M * u0 + c;
	//if (k0.hasVel)
	//	B.segment(row + d, d) = Eigen::VectorXf::Zero(d);

	row += k0.hasVel ? 2*d : 1*d;

	// In-between constraints
	for (int i = 1; i < keyframes.size() - 1; i++)
	{
		Keyframe& ki = keyframes[i];
		ki.t = getNormalizedTime(ki.frame);

		for (int j = 0; j < d; j++)
		{
			float lambda = getLambda(j);
			float delta = getDelta(lambda);

			// std::cout << "Dim: " << j << " Lambda: " << lambda << " Delta: " << delta << std::endl;

			for (int l = 0; l < 4; l++)
			{
				float bj = b(ki.t, delta, lambda, l);
				float bdj = bDot(ki.t, delta, lambda, l);
				float bddj = bDDot(ki.t, delta, lambda, l);
				// std::cout << " Basis" << l << " " << bj << " " << bdj << " " << bddj << std::endl;

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

		row += 3 * d;
	}

	// Boundary condition at t1
	Keyframe& km = keyframes.back();
	km.t = getNormalizedTime(km.frame);

	for (int i = 0; i < d; i++)
	{
		float lambda = getLambda(i);
		float delta = getDelta(lambda);

		for (int j = 0; j < 4; j++)
		{
			int col = getCoeffIdx(keyframes.size() - 2, i, j);
			A(row + i, col) = b(km.t, delta, lambda, j);
			if (km.hasVel)
				A(row + d + i, col) = bDot(km.t, delta, lambda, j);
		}
	}

	u_attrib = km.detail->findFloatTuple(GA_ATTRIB_POINT, "u", 3);
	km.detail->getAttributeAsArray<UT_Vector3F>(u_attrib, km.detail->getPointRange(), displacements);
	Eigen::Map<Eigen::VectorXf> um(displacements.data()->data(), dof);

	B.segment(row, d) = phiT * M * um + c;
	//if (km.hasVel)
	//	B.segment(row+d, d) = Eigen::VectorXf::Zero(d);

	std::cout << "-----------A------------" << std::endl;
	std::cout << A << std::endl;
	std::cout << "-----------b------------" << std::endl;
	std::cout << B << std::endl;

	if (progress.wasInterrupted())
		return;

	int subspaceDim = numCoeffs - numConditions;

	if (subspaceDim == 0)
	{
		// Can be solved exactly
		coefficients = A.colPivHouseholderQr().solve(B);
		// std::cout << coefficients << std::endl;
		return;
	}

	// NOTE: BDCSVD preferred for larger matrix
	Eigen::BDCSVD<Eigen::MatrixXf> svd = A.bdcSvd(Eigen::ComputeFullU | Eigen::ComputeFullV);
	Eigen::MatrixXf U = svd.matrixV()(Eigen::all, Eigen::seq(numCoeffs - subspaceDim, Eigen::last));

	std::cout << svd.info() << std::endl;

	Eigen::VectorXf w0 = svd.solve(B);

	std::cout << w0 << std::endl;

	// NOTE: Is passing in *this the best way to approach functor?
	//WigglyFunctor functor(*this, U, w0, subspaceDim);
	//Eigen::NumericalDiff<WigglyFunctor> numDiff(functor);
	//Eigen::LevenbergMarquardt<Eigen::NumericalDiff<WigglyFunctor>, float> lm(numDiff);

	//// Initial guess
	//Eigen::VectorXf xmin = Eigen::VectorXf::Zero(subspaceDim);

	//int info = lm.minimize(xmin);
	//std::cout << info << std::endl;

	//const gsl_multimin_fdfminimizer_type* T;
	//gsl_multimin_fdfminimizer* s;

	//gsl_vector* x;
	//gsl_vector_set_all(x, 0.0);

	//gsl_multimin_function_fdf objfunc;

	//gsl_objective_data data;
	//data.U = U;
	//data.w0 = w0;

	//objfunc.n = subspaceDim;
	//objfunc.f = gls_objective_f;
	//objfunc.params = &data;

	//x = gsl_vector_alloc(subspaceDim);

	//T = gsl_multimin_fdfminimizer_vector_bfgs;
	//s = gsl_multimin_fdfminimizer_alloc(T, 2);

	//gsl_multimin_fdfminimizer_set(s, &objfunc, x, /*step size*/ 0.01, 1e-4);

	column_vector z;
	z.set_size(subspaceDim);
	for (int i = 0; i < z.size(); i++)
		z(i) = 0.0;

	//std::cout << z << std::endl;

	ObjFunctor objective(*this, U, w0, subspaceDim);

	dlib::find_min_using_approximate_derivatives(
		dlib::bfgs_search_strategy(),
		dlib::objective_delta_stop_strategy(1e-7).be_verbose(),
		objective, z, -1);

	//std::cout << z << std::endl;

	//// Convert dlib to Eigen
	Eigen::VectorXf z_opt = Eigen::VectorXf::Zero(subspaceDim);
	for (int i = 0; i < subspaceDim; i++)
		z_opt(i) = z(i);

	coefficients = w0 + U*z_opt;  // TODO: Can we directly minimize on the member variable?
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

	const float EA = parms.young;

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

	Eigen::IOFormat CommaInitFmt(
		Eigen::StreamPrecision, 
		Eigen::DontAlignCols, ", ", ", ", "", "", " << ", ";");

	std::cout << K.format(CommaInitFmt) << std::endl;
	std::cout << M.format(CommaInitFmt) << std::endl;

	// printf("Determinant: %f\n", K.determinant());

	UT_AutoInterrupt progress2("Finding eigenvalues and eigen vectors.");
	if (progress2.wasInterrupted())
		return;

	// Find eigenvalues and eigenvectors
	// NOTE: Using self adjoint because both K and M are symmetric
	Eigen::GeneralizedEigenSolver<Eigen::MatrixXf> ges;
	ges.compute(K, M);

	eigenValues = ges.eigenvalues().real();
	eigenModes = ges.eigenvectors().real();

	std::cout << "eigen check" << std::endl;
	std::cout << eigenModes << std::endl << std::endl;
	// std::cout << eigenModes.transpose() * M * eigenModes << std::endl;
}