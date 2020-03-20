#include <xerus.h>

#include "ALSres.cpp"
#include <vector>
#include <fstream>
#include <ctime>
#include <queue>
#include <math.h>

#include <boost/regex.hpp>
#include <boost/algorithm/string_regex.hpp>



using namespace xerus;
using xerus::misc::operator<<;


class InternalSolver;
double simpleALS(const TTOperator& _A,const TTOperator& _T, TTTensor& _x, value_t _eps);
void write_to_disc(std::string name, TTOperator &op);
void read_from_disc(std::string name, TTOperator &op);
void read_from_disc(std::string name, TTTensor &x);

/*
 * Main!!
 */
int main() {
	XERUS_LOG(simpleALS,"Begin Tests for ordering ...");
	XERUS_LOG(simpleALS,"---------------------------------------------------------------");
	Index ii,jj;

	size_t d = 50;
	size_t part = 50 ;
	size_t nob = part / 2;
	double eps = 10e-8;
	size_t start_rank = 10;




	xerus::TTTensor phi(std::vector<size_t>(part,2));
	phi = TTTensor::random(std::vector<size_t>(part,2),std::vector<size_t>(part-1,start_rank));


	TTOperator id,F,Finv;
	std::string name = "../data/fock_h2o_shifted" + std::to_string(2*nob) +"_full.ttoperator";
	read_from_disc(name,F);
	XERUS_LOG(info, "Loading shifted and preconditioned Hamiltonian");
	xerus::TTOperator Hs;
	name = "../data/hamiltonian_h2o_shifted" + std::to_string(2*nob) +"_full.ttoperator";
	read_from_disc(name,Hs);

	XERUS_LOG(info,F.ranks());

	double lambda = simpleALS(Hs,F, phi, eps);
	phi.round(10e-14);

	phi.round(eps);
	XERUS_LOG(info, "The ranks of phi are " << phi.ranks() );
	XERUS_LOG(info, "Size of Solution " << phi.datasize() );

	XERUS_LOG(info, "Lambda =  " << std::setprecision(16) << lambda 	);

	return 0;
}

/*
 *
 *
 *
 * Functions
 *
 *
 */
class InternalSolver {
	const size_t d;
	double lambda;
	value_t eps;
	std::vector<Tensor> leftAStack;
	std::vector<Tensor> rightAStack;
	std::vector<Tensor> leftTStack;
	std::vector<Tensor> rightTStack;
	TTTensor& x;
	const TTOperator& A;
	const TTOperator& T;
public:
	size_t maxIterations;

	InternalSolver(const TTOperator& _A,const TTOperator& _T, TTTensor& _x, value_t _eps)
		: d(_x.order()), x(_x), A(_A), T(_T), maxIterations(200), lambda(1.0), eps(_eps)
	{
		leftAStack.emplace_back(Tensor::ones({1,1,1}));
		rightAStack.emplace_back(Tensor::ones({1,1,1}));
		leftTStack.emplace_back(Tensor::ones({1,1,1}));
		rightTStack.emplace_back(Tensor::ones({1,1,1}));
	}


	void push_left_stack(const size_t _position) {
		Index i1, i2, i3, i4, j1 , j2, j3, j4, k1, k2, k3;
		const Tensor &xi = x.get_component(_position);
		const Tensor &Ai = A.get_component(_position);
		const Tensor &Ti = T.get_component(_position);

		Tensor tmpA,tmpT;
		tmpA(i1, i2, i3) = leftAStack.back()(j1, j2, j3)
				*xi(j1, k1, i1)*Ai(j2, k1, k2, i2)*xi(j3, k2, i3);
		leftAStack.emplace_back(std::move(tmpA));
		tmpT(i1, i2, i3) = leftTStack.back()(j1, j2, j3)
				*xi(j1, k1, i1)*Ti(j2, k1, k2, i2)*xi(j3, k2, i3);
		leftTStack.emplace_back(std::move(tmpT));
	}


	void push_right_stack(const size_t _position) {
		Index i1, i2, i3, i4, j1 , j2, j3, j4, k1, k2, k3;
		const Tensor &xi = x.get_component(_position);
		const Tensor &Ai = A.get_component(_position);
		const Tensor &Ti = T.get_component(_position);

		Tensor tmpA,tmpT;
		tmpA(i1, i2, i3) = xi(i1, k1, j1)*Ai(i2, k1, k2, j2)*xi(i3, k2, j3)
				*rightAStack.back()(j1, j2, j3);
		rightAStack.emplace_back(std::move(tmpA));
		tmpT(i1, i2, i3) = xi(i1, k1, j1)*Ti(i2, k1, k2, j2)*xi(i3, k2, j3)
				*rightTStack.back()(j1, j2, j3);
		rightTStack.emplace_back(std::move(tmpT));
	}




	double solve() {
		// Build right stack
		x.move_core(0, true);
		for (size_t pos = d-1; pos > 0; --pos) {
			push_right_stack(pos);
		}

		Index i1, i2, i3, i4, j1 , j2, j3, j4, k1, k2, k3,l1,l2,l3,m1;
		std::vector<double> residuals_ev(10, 1000.0);
		for (size_t itr = 0; itr < maxIterations; ++itr) {
			// Calculate residual and check end condition
			XERUS_LOG(simpleALS, "Iteration: " << itr << " Eigenvalue " << std::setprecision(16) <<  lambda);

			residuals_ev.push_back(lambda);
			if (itr > 1 and std::abs(residuals_ev[residuals_ev.size()-5] - residuals_ev.back()) < 0.00005) {
				return lambda; // We are done!
			}

			// Sweep Left -> Right
			for (size_t corePosition = 0; corePosition < d; ++corePosition) {
				Tensor op, rhs;

				const Tensor &Ai = A.get_component(corePosition);
				const Tensor &Ti = T.get_component(corePosition);

				op(i1, i2, i3, j1, j2, j3) = leftAStack.back()(i1, k1, j1)*Ai(k1, i2, j2, l1)*rightAStack.back()(i3, l1, j3);
				rhs(i1, i2, i3, j1, j2, j3) = leftTStack.back()(i1, k1, j1)*Ti(k1, i2, j2, l1)*rightTStack.back()(i3, l1, j3);

				XERUS_LOG(info,op);
				XERUS_LOG(info,rhs);
				return 0;

				//lambda = xerus::get_smallest_eigenvalue(sol, op);
  	  	//lambda = xerus::get_eigenpair_iterative(x.component(corePosition),op, true,false, 50000, eps);
		  	//XERUS_LOG(info,sol);

				if (corePosition+1 < d) {
					x.move_core(corePosition+1, true);
					push_left_stack(corePosition);
					rightAStack.pop_back();
				}
			}


			// Sweep Right -> Left : only move core and update stacks
			x.move_core(0, true);
			for (size_t corePosition = d-1; corePosition > 0; --corePosition) {
				push_right_stack(corePosition);
				leftAStack.pop_back();
			}


		}
		return lambda;
	}
};

double simpleALS(const TTOperator& _A,const TTOperator& _T, TTTensor& _x,value_t _eps)  {
	InternalSolver solver(_A, _T, _x, _eps);
	return solver.solve();
}


void write_to_disc(std::string name, TTOperator &op){
	std::ofstream write(name.c_str() );
	xerus::misc::stream_writer(write,op,xerus::misc::FileFormat::BINARY);
	write.close();
}

void read_from_disc(std::string name, TTOperator &op){
	std::ifstream read(name.c_str() );
	xerus::misc::stream_reader(read,op,xerus::misc::FileFormat::BINARY);
	read.close();

}

void read_from_disc(std::string name, TTTensor &x){
	std::ifstream read(name.c_str() );
	xerus::misc::stream_reader(read,x,xerus::misc::FileFormat::BINARY);
	read.close();

}
