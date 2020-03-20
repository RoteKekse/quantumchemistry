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
	size_t start_rank = 60;
	value_t factor = -1;



	xerus::TTTensor phi(std::vector<size_t>(part,2));
	phi = TTTensor::random(std::vector<size_t>(part,2),std::vector<size_t>(part-1,start_rank));


	TTOperator id,F,Finv;
	std::string name = "../data/fock_h2o_inv_shifted" + std::to_string(2*nob) +"_full.ttoperator";
	read_from_disc(name,Finv);
	XERUS_LOG(info, "Loading shifted and preconditioned Hamiltonian");
	xerus::TTOperator Hs;
	name = "../data/hamiltonian_h2o_shifted" + std::to_string(2*nob) +"_full.ttoperator";
	read_from_disc(name,Hs);

	XERUS_LOG(info,Finv.ranks());

	double lambda = simpleALS(Hs,factor*Finv, phi, eps);
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

	TTTensor& x;
	const TTOperator& A;
	const TTOperator& T;
public:
	size_t maxIterations;

	InternalSolver(const TTOperator& _A,const TTOperator& _T, TTTensor& _x, value_t _eps)
		: d(_x.order()), x(_x), A(_A), T(_T), maxIterations(200), lambda(1.0), eps(_eps)
	{
		leftAStack.emplace_back(Tensor::ones({1,1,1,1}));
		rightAStack.emplace_back(Tensor::ones({1,1,1,1}));
	}


	void push_left_stack(const size_t _position) {
		Index i1, i2, i3, i4, j1 , j2, j3, j4, k1, k2, k3;
		const Tensor &xi = x.get_component(_position);
		const Tensor &Ai = A.get_component(_position);
		const Tensor &Ti = T.get_component(_position);

		Tensor tmpA;
		tmpA(i1, i2, i3,i4) = leftAStack.back()(j1, j2, j3,j4)
				*xi(j1, k1, i1)*Ti(j2,k1,k2,i2)*Ai(j3, k2, k3, i3)*xi(j4, k3, i4);
		leftAStack.emplace_back(std::move(tmpA));
	}


	void push_right_stack(const size_t _position) {
		Index i1, i2, i3, i4, j1 , j2, j3, j4, k1, k2, k3;
		const Tensor &xi = x.get_component(_position);
		const Tensor &Ai = A.get_component(_position);
		const Tensor &Ti = T.get_component(_position);

		Tensor tmpA;
		tmpA(i1, i2, i3,i4) = xi(i1, k1, j1)*Ti(i2,k1,k2,j2)*Ai(i3, k2, k3, j3)*xi(i4, k3, j4)
				*rightAStack.back()(j1, j2, j3,j4);
		rightAStack.emplace_back(std::move(tmpA));
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
				XERUS_LOG(info,"Core at " << corePosition << " lambda = " << lambda);
				const Tensor &Ai = A.get_component(corePosition);
				const Tensor &Ti = T.get_component(corePosition);



				op(i1, i2, i3, j1, j2, j3) = leftAStack.back()(i1, k1,k2, j1)*Ti(k1, i2, m1, l1)*Ai(k2, m1, j2, l2)*rightAStack.back()(i3, l1,l2, j3);

				//lambda = xerus::get_smallest_eigenvalue(sol, op);
  	  	lambda = xerus::get_smallest_eigenpair(x.component(corePosition),op);
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
