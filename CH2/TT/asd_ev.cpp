#include <xerus.h>


using namespace xerus;
using xerus::misc::operator<<;

double simpleASD(const TTOperator& _A, TTTensor& _x, double _eps, size_t _maxRank);









int main() {
	XERUS_LOG(simpleALS,"Begin Tests for ordering ...");
	XERUS_LOG(simpleALS,"---------------------------------------------------------------");


	size_t d = 26; // 16 electron, 8 electron pairs
	size_t nob = 13;
	double eps = 0.000000001;
	size_t start_rank = 50;
	size_t max_rank = 10;
	size_t wsize = 5;
	XERUS_LOG(simpleMALS,"Load Operator ...");


	xerus::TTTensor phi = xerus::TTTensor::random(std::vector<size_t>(d,2),std::vector<size_t>(d - 1,start_rank));
	xerus::TTOperator op;
	std::string name = "../hamiltonian_CH2_" + std::to_string(d) +"_full.ttoperator";
	std::ifstream read(name.c_str());
	misc::stream_reader(read,op,xerus::misc::FileFormat::BINARY);
	read.close();
	auto id = TTOperator::identity(op.dimensions);
	op = 12 *id+op;
	double lambda = simpleASD(op, phi, eps, max_rank);
	  //double lambda = simpleALS(op, phi);



		phi.round(10e-14);
		XERUS_LOG(info, "The ranks of op are " << op.ranks() );
		XERUS_LOG(info, "The ranks of phi are " << phi.ranks() );
		XERUS_LOG(info, "lambda = " << lambda + 8.80146457125193 );
		XERUS_LOG(info, "err = " << std::abs(lambda + 8.80146457125193 + 76.256624));
		XERUS_LOG(info, "lambda op = " << lambda  );

}



class InternalSolver {
	const size_t d;
	double lambda;
	double eps;
	size_t maxRank;

	TTTensor& x;
	const TTOperator& A;
	TTOperator AmlI;
public:
	size_t maxIterations;

	InternalSolver(const TTOperator& _A, TTTensor& _x, double _eps, size_t _maxRank)
		: d(_x.degree()), x(_x), A(_A),  maxIterations(200), lambda(1.0), eps(_eps), maxRank(_maxRank)
	{
		AmlI = TTOperator::identity(A.dimensions);
	}


	Tensor get_left(const size_t _position) {
		Index i1, i2, i3, j1 , j2, j3, k1, k2;
		auto tmp = Tensor::ones({1,1,1});
		for ( size_t i = 0; i < _position; ++i){
			const Tensor &xi = x.get_component(i);
			const Tensor &Ai = AmlI.get_component(i);
			tmp(i1,i2,i3) = tmp(j1,j2,j3) * xi(j1,k1,i1) * Ai(j2,k1,k2,i2) * xi(j3,k2,i3);
		}
		return tmp;
	}


	Tensor get_right(const size_t _position) {
		Index i1, i2, i3, j1 , j2, j3, k1, k2;
		auto tmp = Tensor::ones({1,1,1});
		for ( size_t i = d - 1; i > _position; --i){
			const Tensor &xi = x.get_component(i);
			const Tensor &Ai = AmlI.get_component(i);
			tmp(i1,i2,i3) =  xi(i1,k1,j1) * Ai(i2,k1,k2,j2) * xi(i3,k2,j3) * tmp(j1,j2,j3);
		}
		return tmp;
	}


	double solve() {
		// Build right stack
		x.move_core(0, true);

		Index i1, i2, i3, i4, j1 , j2, j3, j4, k1, k2, k3;
		Index a1, a2, a3, a4, a5, r1, r2;
		std::vector<double> residuals_ev(10, 1000.0);
		std::vector<double> residuals(10, 1000.0);
		XERUS_LOG(info,"A = " << A.ranks());

		for (size_t itr = 0; itr < maxIterations; ++itr) {

			// Calculate residual and check end condition
			residuals_ev.push_back(lambda);
			//residuals.push_back(calc_residual_norm());
			if (itr > 1 and std::abs(residuals_ev[residuals_ev.size()-10] - residuals_ev.back()) < eps) {
				return lambda; // We are done!
			}

			// Sweep Left -> Right
			for (size_t corePosition = 0; corePosition < d; ++corePosition) {
				x.move_core(corePosition);
				Tensor op, rhs, U, S, Vt;
				//XERUS_LOG(simpleMALS, "Iteration: " << itr  << " core: " << corePosition  << " Eigenvalue " << std::setprecision(16) <<  lambda);
				auto id = TTOperator::identity(A.dimensions);

				AmlI = A - lambda * id;

				const Tensor &Ai = AmlI.get_component(corePosition);
				const Tensor &xi = x.get_component(corePosition);
				const Tensor left = get_left(corePosition);
				const Tensor right = get_right(corePosition);
//				XERUS_LOG(info,"Ai    = " << Ai.dimensions);
//				XERUS_LOG(info,"xi    = " << xi.dimensions);
//				XERUS_LOG(info,"left  = " << left.dimensions);
//				XERUS_LOG(info,"right = " << right.dimensions);


				Tensor Pxi;
				Pxi (i1,i2,i3) = left(i1,k1,j1) * xi(j1,k2,j2) * Ai(k1,k2,i2,k3) * right(i3,k3,j2);
				//calculate alpha k
				auto tmp = Tensor::ones({1,1,1});
				auto tmp2 = Tensor::ones({1,1,1});
				auto tmp3 = Tensor::ones({1,1,1});
				auto ones = Tensor::ones({1,1,1});
				for (size_t i = 0; i < d; ++i){
					const Tensor &xi = x.get_component(i);
					const Tensor &Ai = A.get_component(i);
					const Tensor &Ai1 = AmlI.get_component(i);
					if (i == corePosition){
						tmp(i1,i2,i3) = tmp(j1,j2,j3) * Pxi(j1,k1,i1) * Ai1(j2,k1,k2,i2) * xi(j3,k2,i3);
						tmp2(i1,i2,i3) = tmp2(j1,j2,j3) * Pxi(j1,k1,i1) * Ai(j2,k1,k2,i2) * Pxi(j3,k2,i3);
					} else {
						tmp(i1,i2,i3) = tmp(j1,j2,j3) * xi(j1,k1,i1) * Ai1(j2,k1,k2,i2) * xi(j3,k2,i3);
						tmp2(i1,i2,i3) = tmp2(j1,j2,j3) * xi(j1,k1,i1) * Ai(j2,k1,k2,i2) * xi(j3,k2,i3);
					}
				}
				tmp() = tmp(j1,j2,j3) * ones(j1,j2,j3);
				tmp2() = tmp2(j1,j2,j3) * ones(j1,j2,j3);
				value_t ak = tmp[0] / tmp2[0];

				x.set_component(corePosition,xi - ak * Pxi);
				x /= x.frob_norm();
				for (size_t i = 0; i < d; ++i){
					const Tensor &xi = x.get_component(i);
					const Tensor &Ai = A.get_component(i);
					tmp3(i1,i2,i3) = tmp3(j1,j2,j3) * xi(j1,k1,i1) * Ai(j2,k1,k2,i2) * xi(j3,k2,i3);
				}
				tmp3() = tmp3(j1,j2,j3) * ones(j1,j2,j3);
				lambda = tmp3[0];
				XERUS_LOG(simpleALS,"CP =  " << corePosition << " l: " << lambda << " l corr " << lambda - 28.1930439210 - 12 <<  " err = " << std::abs(lambda -12 - 28.1930439210 +38.979392539208) );// << " Residual = " << calc_residual_norm());

			}
			XERUS_LOG(simpleALS, "Iteration: " << itr  <<  " EV " << std::setprecision(10) <<  lambda - 28.1930439210 - 12 << " EV2 = " << lambda - 12 << " err = " << std::abs(lambda -12 - 28.1930439210 +38.979392539208));// << " Residual = " << calc_residual_norm());
			XERUS_LOG(info,"x = " << x.ranks());

		}
		return lambda;
	}
};

double simpleASD(const TTOperator& _A, TTTensor& _x, double _eps, size_t _maxRank)  {
	InternalSolver solver(_A, _x, _eps, _maxRank);
	return solver.solve();
}
