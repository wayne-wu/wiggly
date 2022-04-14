#include "GU/GU_PrimTetrahedron.h"
#include "UT/UT_Interrupt.h"
#include "UT/UT_UniquePtr.h"

#include "Wiggly.h"

#include "Eigen/Eigenvalues"
#include "Eigen/SVD"
#include "Eigen/QR"

#include <dlib/optimization.h>

#include <iostream>

#define DEBUG 0
#define DEBUG_EIGEN 0
#define CANTOR(a, b) (a + b) * (a + b + 1) / 2 + a


using namespace HDK_Wiggly;


typedef dlib::matrix<double, 0, 1> column_vector;

/*
Functor for the energy objective function using dlib minimization
*/
struct ObjFunctor
{
	ObjFunctor(Wiggly& obj, const MatX& U, const VecX& w0, const int& dim)
		: wiggly(obj), U(U), w0(w0), subDim(dim) {}

	double operator()(const column_vector& v) const
	{
		if (wiggly.getProgress().wasInterrupted())
			return 0.0;

		// TODO: Use Eigen::Map
		VecX z = VecX::Zero(subDim);
		for (int i = 0; i < subDim; i++)
			z(i) = v(i);

		return wiggly.totalEnergy(w0 + U * z);
	}

	Wiggly& wiggly;
	const MatX& U;
	const VecX& w0;
	const int subDim;
};

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
scalar Wiggly::b(const float t, const scalar delta, const scalar lambda, int i)
{
	int sign1 = i <= 1 ? -1 : 1;
	int sign2 = i % 2 == 0 ? 1 : -1;

	// delta == 0 && lambda == 0
	if (abs(delta) < parms.eps && abs(lambda) < parms.eps)
		return pow(t, i);

	// delta != 0 && lambda == 0
	if (abs(delta) > parms.eps && abs(lambda) < parms.eps)
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

	// delta == 0 && lambda > 0
	if (abs(delta) < parms.eps && lambda > parms.eps)
	{
		scalar n = sqrt(abs(lambda));
		switch (i) {
		case 0:
			return cos(n * t);
		case 1:
			return sin(n * t);
		case 2:
			return t * cos(n * t);
		case 3:
			return t * sin(n * t);
		default:
			return pow(t, i);
		}
	}

	scalar tmp = delta * delta - lambda;

	// delta == 0 && lambda < 0 or n == 0 && lambda > 0
	if ((abs(delta) < parms.eps && lambda < -parms.eps) || sqrt(abs(tmp)) < parms.eps && lambda > parms.eps)
	{
		scalar e = sqrt(abs(lambda));
		switch (i) {
		case 0:
			return exp(-e * t);
		case 1:
			return exp(e * t);
		case 2:
			return t * exp(-e * t);
		case 3:
			return t * exp(e * t);
		default:
			return pow(t, i);
		}
	}

	if (tmp > parms.eps)
		return exp(t * (sign1 * delta + sign2 * sqrt(tmp)));
	else if (sign2 == 1) // b1 & b3
		return exp(sign1 * t * delta) * cos(t * sqrt(-tmp));
	else // b2 & b4
		return exp(sign1 * t * delta) * sin(t * sqrt(-tmp));
}

/*
First derivative of the basis function
*/
scalar Wiggly::bDot(const float t, const scalar delta, const scalar lambda, int i)
{
	int sign1 = i <= 1 ? -1 : 1;
	int sign2 = i % 2 == 0 ? 1 : -1;

	if (abs(delta) < parms.eps && abs(lambda) < parms.eps)
		return i == 0 ? 0 : i * pow(t, i - 1);

	if (abs(delta) > parms.eps && abs(lambda) < parms.eps)
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

	// delta == 0 && lambda > 0
	if (abs(delta) < parms.eps && lambda > parms.eps)
	{
		scalar n = sqrt(abs(lambda));
		switch (i) {
		case 0:
			return -n * sin(n * t);
		case 1:
			return n * cos(n * t);
		case 2:
			return - t * n * sin(n * t) + cos(n * t);
		case 3:
			return t * n * cos(n * t) + sin(n * t);
		default:
			return pow(t, i);
		}
	}

	scalar tmp = delta * delta - lambda;

	// delta == 0 && lambda < 0 or n == 0 && lambda > 0
	if ((abs(delta) < parms.eps && lambda < -parms.eps) || sqrt(abs(tmp)) < parms.eps && lambda > parms.eps)
	{
		scalar e = sqrt(abs(lambda));
		switch (i) {
		case 0:
			return -e*exp(-e * t);
		case 1:
			return e*exp(e * t);
		case 2:
			return -t * e * exp(-e * t) + exp(-e * t);
		case 3:
			return t * e * exp(e * t) + exp(e * t);
		default:
			return pow(t, i);
		}
	}

	if (tmp > parms.eps)
	{
		scalar expr = sign1 * delta + sign2 * sqrt(tmp);
		return exp(t * expr) * expr;
	}
	else
	{
		scalar expr = sqrt(-tmp);
		if (sign2 == 1)
			return exp(sign1 * t * delta) * ((sign1 * delta * cos(t * expr)) - (expr * sin(t * expr)));
		else
			return -exp(sign1 * t * delta) * (-(expr * cos(t * expr)) - (sign1 * delta * sin(t * expr)));
	}
}

/*
Second derivative of the basis function
*/
scalar Wiggly::bDDot(const float t, const scalar delta, const scalar lambda, int i)
{
	int sign1 = i <= 1 ? -1 : 1;
	int sign2 = i % 2 == 0 ? 1 : -1;

	if (abs(delta) < parms.eps && abs(lambda) < parms.eps)
		return i <= 1 ? 0 : i * (i - 1) * pow(t, i - 2);

	if (abs(delta) > parms.eps && abs(lambda) < parms.eps)
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

	if (abs(delta) < parms.eps && lambda > parms.eps)
	{
		scalar n = sqrt(abs(lambda));
		switch (i) {
		case 0:
			return -n * n * cos(n * t);
		case 1:
			return -n * n * sin(n * t);
		case 2:
			return -n * (t*n*cos(n*t) + sin(n*t) + sin(n * t));
		case 3:
			return n * (-t*n*sin(n*t) + cos(n*t) + cos(n * t));
		default:
			return pow(t, i);
		}
	}

	scalar tmp = delta * delta - lambda;

	// delta == 0 && lambda < 0 or n == 0 && lambda > 0
	if ((abs(delta) < parms.eps && lambda < -parms.eps) || sqrt(abs(tmp)) < parms.eps && lambda > parms.eps)
	{
		scalar e = sqrt(abs(lambda));
		switch (i) {
		case 0:
			return e*e * exp(-e * t);
		case 1:
			return e*e * exp(e * t);
		case 2:
			return -e * (-t * e * exp(-e * t) + exp(-e * t) + exp(-e * t));
		case 3:
			return e * (t * e * exp(e * t) + exp(e*t) + exp(e * t));
		default:
			return pow(t, i);
		}
	}

	if (tmp > parms.eps)
	{
		scalar expr = sign1 * delta + sign2 * sqrt(tmp);
		return exp(t * expr) * expr * expr;
	}
	else
	{
		scalar expr = sqrt(-tmp);
		scalar dsqExpr = 2 * delta * delta - lambda;
		if (sign2 == 1)
			return exp(sign1 * t) * ((dsqExpr * cos(t * expr)) + (2 * delta * expr * (-sign1) * sin(t * expr)));
		else
			return sign1 * exp(sign1 * t * delta) * ((2 * delta * expr * cos(t * expr)) + (sign1 * dsqExpr * sin(t * expr)));
	}
}

/*
Compute the wiggly spline value
*/
scalar Wiggly::wiggly(const float t, const int d, const scalar delta, const scalar lambda, const VecX& coeffs)
{
	int k = getSegmentIdx(t);

	scalar sum = 0;
	for(int l = 0; l < 4; l++)
		sum += coeffs[getCoeffIdx(k, d, l)] * b(t, delta, lambda, l);

	return sum - getC(lambda);
}	

/*
Compute the first derivative of the wiggly spline
*/
scalar Wiggly::wigglyDot(const float t, const int d, const scalar delta, const scalar lambda, const VecX& coeffs)
{
	int k = getSegmentIdx(t);

	scalar sum = 0;
	for (int l = 0; l < 4; l++)
		sum += coeffs[getCoeffIdx(k, d, l)] * bDot(t, delta, lambda, l);

	return sum;
}

/*
Compute the first derivative of the wiggly spline
*/
scalar Wiggly::wigglyDDot(const float t, const int d, const scalar delta, const scalar lambda, const VecX& coeffs)
{
	int k = getSegmentIdx(t);

	scalar sum = 0;
	for (int l = 0; l < 4; l++)
		sum += coeffs[getCoeffIdx(k, d, l)] * bDDot(t, delta, lambda, l);

	return sum;
}

VecX Wiggly::u(const float f)
{
	float t = getNormalizedTime(f);
	return u(t, coefficients);
}

/*
Compute the displacement vector
*/
VecX Wiggly::u(const float t, const VecX& coeffs)
{
	VecX out = VecX::Zero(getDof());
	for (int i = 0; i < parms.d; i++)
	{
		scalar lambda = getLambda(i);
		scalar delta = getDelta(lambda);
		out += wiggly(t, i, delta, lambda, coeffs) * eigenModes.col(i);
	}
	return out;
}

/*
Compute the velocity vector
*/
VecX Wiggly::uDot(const float t, const VecX& coeffs)
{
	VecX out = VecX::Zero(getDof());
	for (int i = 0; i < parms.d; i++)
	{
		scalar lambda = getLambda(i);
		scalar delta = getDelta(lambda);
		out += wigglyDot(t, i, delta, lambda, coeffs) * eigenModes.col(i);
	}
	return out;
}

scalar Wiggly::totalEnergy(const VecX& c)
{
	scalar d = dynamicsEnergy(c);
	scalar k = keyframeEnergy(c);
	// std::cout << "d: " << d << "   k: " << k << std::endl;
	return k + parms.physical * d;
}

/*
Compute the energy based on constraints
*/
scalar Wiggly::keyframeEnergy(const VecX& coeffs)
{
	scalar total = 0;
	for (Keyframe& k : keyframes)
	{
		VecX uPos = u(k.t, coeffs);
		VecX uVel = uDot(k.t, coeffs);

		GA_ROHandleV3D v_h(k.detail, GA_ATTRIB_POINT, "v");

		scalar posDiff = 0, velDiff = 0;
		for (int j = 0; j < k.points.size(); ++j)
		{
			int ptIdx = k.points[j];
			if (ptIdx < 0) 
				continue;

			if (k.hasPos) 
				posDiff += k.u[ptIdx].distance2(
					UT_Vector3D(uPos(3 * ptIdx), uPos(3 * ptIdx + 1), uPos(3 * ptIdx + 2)));

			if (k.hasVel)
				velDiff += v_h.get(k.detail->pointOffset(ptIdx)).distance2(
					UT_Vector3D(uVel(3 * ptIdx), uVel(3 * ptIdx + 1), uVel(3 * ptIdx + 2)));
		}
		total += posDiff * parms.cA;
		total += velDiff * parms.cB;
	}
	return 0.5 * total;
}

/*
The integrand for the energy minimizing integrand
*/
scalar Wiggly::integrand(const float t, const int d, const scalar delta, const scalar lambda, const VecX& coeffs)
{
	scalar tmp = wigglyDDot(t, d, delta, lambda, coeffs) + 
		2 * delta * wigglyDot(t, d, delta, lambda, coeffs) + 
		lambda * wiggly(t, d, delta, lambda, coeffs) + parms.g;
	return tmp * tmp;
}

/*
Evaluate the energy of a wiggly spline based on the integral
*/
scalar Wiggly::integralEnergy(const int d, const scalar delta, const scalar lambda, const VecX& coeffs)
{	
	int intervals = 100;

	float a = keyframes.front().t;  // this should be 0 after normalizing
	float b = keyframes.back().t;  // this should be 1 after normalizing

	float stepSize = 1.0 / intervals;

	float t = a;
	scalar sum = integrand(a, d, delta, lambda, coeffs) + integrand(b, d, delta, lambda, coeffs);
	
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
scalar Wiggly::dynamicsEnergy(const VecX& coeffs)
{
	scalar total = 0;
	for (int i = 0; i < parms.d; ++i)
	{
		scalar lambda = getLambda(i);
		scalar delta = getDelta(lambda);
		scalar e = integralEnergy(i, delta, lambda, coeffs);
		total += e;
	}

	return total;
}

/*
Compute the wiggly splines' coefficients based on the sparse keyframes.
This should be recomputed when the mesh changes or the keyframes change.
*/
void Wiggly::compute() {

	progress = UTmakeUnique<UT_AutoInterrupt>("Assembling condition matrix.");
	if (progress->wasInterrupted())
		return;

	int dof = getDof();
	int d = parms.d;

	int numCoeffs = getTotalNumCoeffs();
	int numConditions = 4 * d;
	for (int i = 1; i < keyframes.size() - 1; i++)
		numConditions += keyframes[i].hasVel ? 2 * d : 3 * d;

	MatX A = MatX::Zero(numConditions, numCoeffs);
	VecX B = VecX::Zero(numConditions);
	VecX c = VecX::Zero(d);
	MatX phiTM = eigenModes(Eigen::all, Eigen::seqN(0,d)).transpose()*M;  // This should be d x 3n

#if DEBUG_EIGEN
	MatX phi = eigenModes(Eigen::all, Eigen::seqN(0, d));
	std::cout << "==============Eigen Check===================" << std::endl;
	std::cout << phi.transpose() * M * phi << std::endl << std::endl;
	std::cout << phi * phi.transpose() * M << std::endl;
#endif

	int row = 0; // The row correlates to the number of linear conditions

	// Boundary condition at t0
	Keyframe& k0 = keyframes.front();
	k0.t = getNormalizedTime(k0.frame);  // TODO: Where's the best place to do this?
	for (int i = 0; i < d; i++)
	{
		scalar lambda = getLambda(i);
		scalar delta = getDelta(lambda);
		c(i) = getC(lambda);

		for (int j = 0; j < 4; j++)
		{
			int col = getCoeffIdx(0, i, j);
			A(row + i, col) = b(k0.t, delta, lambda, j);
			if (k0.hasVel)
				A(row + d + i, col) = bDot(k0.t, delta, lambda, j);
		}
	}

	Eigen::Map<VecX> u0(k0.u.data()->data(), dof);

	B.segment(row, d) = phiTM * u0 + c;
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
			scalar lambda = getLambda(j);
			scalar delta = getDelta(lambda);

			for (int l = 0; l < 4; l++)
			{
				scalar bj = b(ki.t, delta, lambda, l);
				scalar bdj = bDot(ki.t, delta, lambda, l);
				scalar bddj = bDDot(ki.t, delta, lambda, l);

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
		scalar lambda = getLambda(i);
		scalar delta = getDelta(lambda);

		for (int j = 0; j < 4; j++)
		{
			int col = getCoeffIdx(keyframes.size() - 2, i, j);
			A(row + i, col) = b(km.t, delta, lambda, j);
			if (km.hasVel)
				A(row + d + i, col) = bDot(km.t, delta, lambda, j);
		}
	}

	Eigen::Map<VecX> um(km.u.data()->data(), dof);

	B.segment(row, d) = phiTM * um + c;
	//if (km.hasVel)
	//	B.segment(row+d, d) = Eigen::VectorXf::Zero(d);

#if DEBUG
	std::cout << "===========A============" << std::endl;
	Eigen::IOFormat CommaInitFmt(
		Eigen::StreamPrecision,
		Eigen::DontAlignCols, ", ", ", ", "", "", " << ", ";");
	std::cout << A.format(CommaInitFmt) << std::endl;
	std::cout << "Det(A) = " << A.determinant() << std::endl;
	std::cout << "===========b============" << std::endl;
	std::cout << B << std::endl;
#endif

	if (progress->wasInterrupted())
		return;

	int subspaceDim = numCoeffs - numConditions;

	if (subspaceDim == 0)
	{
		// Can be solved exactly
		coefficients = A.colPivHouseholderQr().solve(B);
		return;
	}

	// NOTE: BDCSVD preferred for larger matrix
	Eigen::BDCSVD<MatX> svd = A.bdcSvd(Eigen::ComputeFullU | Eigen::ComputeFullV);
	MatX U = svd.matrixV()(Eigen::all, Eigen::seq(numCoeffs - subspaceDim, Eigen::last));

	VecX w0 = svd.solve(B);

	progress.reset();
	progress = UTmakeUnique<UT_AutoInterrupt>("Running Optimization");
	if (progress->wasInterrupted())
		return;

	// TODO: What should the starting value be?
	column_vector z;
	z.set_size(subspaceDim);
	for (int i = 0; i < z.size(); i++)
		z(i) = 1.0;

	ObjFunctor objective(*this, U, w0, subspaceDim);

	dlib::find_min_using_approximate_derivatives(
		dlib::bfgs_search_strategy(),
		dlib::objective_delta_stop_strategy(parms.eps),
		objective, z, -1);

	//// Convert dlib to Eigen
	VecX z_opt = VecX::Zero(subspaceDim);
	for (int i = 0; i < subspaceDim; i++)
		z_opt(i) = z(i);

	coefficients = w0 + U * z_opt;  // TODO: Can we directly minimize on the member variable?

	progress.reset();
}

/*
Compute eigenvalues and eigenvectors based on stiffness and mass matrices
as part of the precompute step. This only needs to be recomputed if the mesh changes.
*/
void Wiggly::preCompute() {

  progress = UTmakeUnique<UT_AutoInterrupt>("Assembling stiffness and mass matrices.");
	
	int dof = getDof();

	/*
	Compute the global stiffness matrix of the tetrahedral mesh.
	Adopted from 11.2 in Finite Element Method in Engineering (6th Edition)
	*/
	K = MatX::Zero(dof, dof);

	/*
	Compute the global mass matrix of the tetrahedral mesh.
	Adopted from 12.2.8 in Finite Element Method in Engineering (6th Edition)
	*/
	M = MatX::Zero(dof, dof);

	for (GA_Iterator it(mesh->getPrimitiveRange()); !it.atEnd(); ++it)
	{
		if (progress->wasInterrupted())
			return;

		const GU_PrimTetrahedron* tet = (const GU_PrimTetrahedron*)mesh->getPrimitive(*it);

		int subrows[12];

		Eigen::Matrix4d m;
		for (int i = 0; i < 4; i++)
		{
			int idx = (i + 1) % 4; // shift index by 1 so that it's 1->2->3->0
			const UT_Vector3F& p = tet->getPos3(idx);
			m(i, 0) = 1;
			m(i, 1) = p.x();
			m(i, 2) = p.y();
			m(i, 3) = p.z();

			const GA_Index ptIdx = tet->getPointIndex(idx);
			subrows[3*i + 0] = 3 * ptIdx + 0;
			subrows[3*i + 1] = 3 * ptIdx + 1;
			subrows[3*i + 2] = 3 * ptIdx + 2;
		}

		scalar V = m.determinant() / 6.; // tet->calcVolume(UT_Vector3(0, 0, 0));

		m = m.transpose().inverse().eval();
		m = 6 * V * m;

		auto a = m.col(0);
		auto b = m.col(1);
		auto c = m.col(2);
		auto d = m.col(3);

		MatX B = MatX::Zero(6,12);

		B << b[0], 0, 0, b[1], 0, 0, b[2], 0, 0, b[3], 0, 0,
					0, c[0], 0, 0, c[1], 0, 0, c[2], 0, 0, c[3], 0,
					0, 0, d[0], 0, 0, d[1], 0, 0, d[2], 0, 0, d[3],
					c[0], b[0], 0, c[1], b[1], 0, c[2], b[2], 0, c[3], b[3], 0,
					0, d[0], c[0], 0, d[1], c[1], 0, d[2], c[2], 0, d[3], c[3],
					d[0], 0, b[0], d[1], 0, b[1], d[2], 0, b[2], d[3], 0, b[3];

		B *= 1.0/(6.0*V);

		MatX D = MatX::Zero(6, 6);
		scalar v = parms.poisson;
		scalar E = parms.young;

		D.block(0, 0, 3, 3).setConstant(v);
		D(0, 0) = D(1, 1) = D(2, 2) = (1 - v);
		D(3, 3) = D(4, 4) = D(5, 5) = (1 - 2 * v) * 0.5;
		D *= E / ((1 + v)*(1 - 2*v));

		K(subrows, subrows) += V * B.transpose() * D * B;

		MatX subM = MatX::Zero(12, 12);
		subM.diagonal().setConstant(2);
		for (int i : { 3, 6, 9 })
		{
			subM.diagonal(i).setConstant(1);
			subM.diagonal(-i).setConstant(1);
		}
		subM *= parms.p*V/20.0;

		M(subrows, subrows) += subM;
	}

#if DEBUG
	Eigen::IOFormat CommaInitFmt(
		Eigen::StreamPrecision,
		Eigen::DontAlignCols, ", ", ", ", "", "", " << ", ";");
	std::cout << K << std::endl;
	std::cout << M << std::endl;
#endif

	progress = UTmakeUnique<UT_AutoInterrupt>("Finding eigenvalues and eigen vectors.");
	if (progress->wasInterrupted())
		return;

	// Find eigenvalues and eigenvectors
	// NOTE: Using self adjoint because K is symmetric and M is positive definite
	// Self adjoint is much faster and more accurate
	Eigen::GeneralizedSelfAdjointEigenSolver<MatX> ges;
	ges.compute(K, M);

	// NOTE: Eigen values should be from smallest to largest
	eigenValues = ges.eigenvalues().real();
	eigenModes = ges.eigenvectors().real();

	progress.reset();
}