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
double simpleMALS(const TTOperator& _A,const TTOperator& _T, TTTensor& _x, double _eps, size_t _maxRank);
void write_to_disc(std::string name, TTOperator &op);
void read_from_disc(std::string name, TTOperator &op);
void read_from_disc(std::string name, TTTensor &x);

/*
 * Main!!
 */
int main() {
	XERUS_LOG(simpleASD,"---------------------------------------------------------------");
	Index ii,jj;

	size_t d = 50;
	size_t part = 50 ;
	size_t nob = part / 2;
	double eps = 10e-6;
	size_t start_rank = 4;
	size_t max_rank = 60;

	TTOperator id,Finv,Fsqinv(std::vector<size_t>(4*nob,2)),Fsq(std::vector<size_t>(4*nob,2)),test;
	std::string name = "../data/fock_h2o_inv_shifted" + std::to_string(2*nob) +"_full.ttoperator";
	read_from_disc(name,Finv);
	id = xerus::TTOperator::identity(std::vector<size_t>(4*nob,2));
	Finv = (-1)*Finv;
	Finv.round(1);

	xerus::TTTensor phi(std::vector<size_t>(part,2));
	read_from_disc("../data/start_vector_H2O_rank_60.tttensor",phi);


	xerus::TTOperator op;
	name = "../data/hamiltonian_H2O_" + std::to_string(d) +"_full_2.ttoperator";
	//name = "../data/Fock_inv_sqr_H2O_" + std::to_string(2*nob) +".ttoperator";
	std::ifstream read(name.c_str());
	misc::stream_reader(read,op,xerus::misc::FileFormat::BINARY);
	read.close();

	double lambda = simpleMALS(op, Finv, phi, eps, max_rank);
  //double lambda = simpleALS(op, phi);
	phi.round(10e-14);

	std::string name2 = "../data/eigenvector_H2O_" + std::to_string(d) +"_" + std::to_string(lambda) +"_3.tttensor";
	std::ofstream write(name2.c_str() );
	xerus::misc::stream_writer(write,phi,xerus::misc::FileFormat::BINARY);
	write.close();





	XERUS_LOG(simpleASD, "The ranks of phi are " << phi.ranks() );
	XERUS_LOG(simpleASD, "Size of Solution " << phi.datasize() );

	XERUS_LOG(simpleASD, "Lambda =  " << std::setprecision(16) << lambda 	);
	//XERUS_LOG(info, "Lambda Error =  " << std::setprecision(16) << std::abs(lambda - 52.4190597253 +76.25663396));

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
	double eps;
	size_t maxRank;
	std::vector<Tensor> leftAStack;
	std::vector<Tensor> leftA1Stack;
	std::vector<Tensor> rightAStack;
	std::vector<Tensor> rightA1Stack;
	std::vector<Tensor> leftxStack;
	std::vector<Tensor> rightxStack;
	Tensor last_grad;

	TTTensor& x;
	const TTOperator& A;
	const TTOperator& T;
public:
	size_t maxIterations;

	InternalSolver(const TTOperator& _A,const TTOperator& _T, TTTensor& _x, double _eps, size_t _maxRank)
		: d(_x.order()), x(_x), A(_A), T(_T), maxIterations(50), lambda(1.0), eps(_eps), maxRank(_maxRank)
	{
		leftAStack.emplace_back(Tensor::ones({1,1,1,1}));
		rightAStack.emplace_back(Tensor::ones({1,1,1,1}));
		leftA1Stack.emplace_back(Tensor::ones({1,1,1}));
		rightA1Stack.emplace_back(Tensor::ones({1,1,1}));
		leftxStack.emplace_back(Tensor::ones({1,1,1}));
		rightxStack.emplace_back(Tensor::ones({1,1,1}));
	}


	void push_left_stack(const size_t _position) {
		Index i1, i2, i3, i4, j1 , j2, j3, j4,k1, k2,k3;
		const Tensor &xi = x.get_component(_position);
		const Tensor &Ai = A.get_component(_position);
		const Tensor &Ti = T.get_component(_position);

		Tensor tmpA,tmpA1,tmpx;
		tmpA(i1, i2, i3,i4) = leftAStack.back()(j1, j2, j3,j4)
				*xi(j1, k1, i1)*Ti(j2,k1,k2,i2)*Ai(j3, k2, k3, i3)*xi(j4, k3, i4);
		leftAStack.emplace_back(std::move(tmpA));
		tmpA1(i1, i2, i3) = leftA1Stack.back()(j1, j2, j3)
				*xi(j1, k1, i1)*Ai(j2, k1, k2, i2)*xi(j3, k2, i3);
		leftA1Stack.emplace_back(std::move(tmpA1));
		tmpx(i1, i2, i3) = leftxStack.back()(j1, j2, j3)
						*xi(j1, k1, i1)*Ti(j2,k1,k2,i2)*xi(j3, k2, i3);
		leftxStack.emplace_back(std::move(tmpx));
	}


	void push_right_stack(const size_t _position) {
		Index i1, i2, i3, i4, j1 , j2, j3, j4,k1, k2,k3;
		const Tensor &xi = x.get_component(_position);
		const Tensor &Ai = A.get_component(_position);
		const Tensor &Ti = T.get_component(_position);

		Tensor tmpA,tmpA1,tmpx;
		tmpA(i1, i2, i3,i4) = xi(i1, k1, j1)*Ti(i2,k1,k2,j2)*Ai(i3, k2, k3, j3)*xi(i4, k3, j4)
				*rightAStack.back()(j1, j2, j3,j4);
		rightAStack.emplace_back(std::move(tmpA));
		tmpA1(i1, i2, i3) = xi(i1, k1, j1)*Ai(i2, k1, k2, j2)*xi(i3, k2, j3)
				*rightA1Stack.back()(j1, j2, j3);
		rightA1Stack.emplace_back(std::move(tmpA1));
		tmpx(i1, i2, i3) = xi(i1, k1, j1)*Ti(i2, k1, k2, j2)*xi(i3, k2, j3)
						*rightxStack.back()(j1, j2, j3);
		rightxStack.emplace_back(std::move(tmpx));
	}



	double solve() {
		// Build right stack
		x.move_core(0, true);
		for (size_t pos = d-1; pos > 0; --pos) {
			push_right_stack(pos);
		}

	  std::ofstream outfile;
		std::string out_name = "/homes/numerik/goette/Documents/jupyter_examples/Paper/Preconditioning/data/ASD_rank_" + std::to_string(maxRank) +"_cpu.csv";
		outfile.open(out_name);
		outfile.close();
	  clock_t begin_time,global_time = clock();
	  value_t stack_time = 0,solving_time = 0;

		Index i1, i2, i3, i4, j1 , j2, j3, j4, k1, k2, k3,l1,l2,l3;
		Index a1, a2, a3, a4, a5, r1, r2;
		std::vector<double> residuals_ev(10, 1000.0);
		std::vector<double> residuals(10, 1000.0);
		XERUS_LOG(simpleASD,"A = " << A.ranks());
		std::vector<value_t> result;

		for (size_t itr = 0; itr < maxIterations; ++itr) {
//			if (itr % 5 == 0 and itr != 0)
//				maxRank = 250 > maxRank + 10 ? maxRank + 10 : 250;
			stack_time = 0,solving_time = 0;


			// Calculate residual and check end condition
			residuals_ev.push_back(lambda);
			if (itr > 1 and std::abs(residuals_ev[residuals_ev.size()-10] - residuals_ev.back()) < eps) {
				XERUS_LOG(simpleASD, residuals_ev[residuals_ev.size()-10]);
				XERUS_LOG(simpleASD, residuals_ev.back());
				XERUS_LOG(simpleASD, eps);
				XERUS_LOG(simpleASD, "Done! Residual decreased to residual "  << std::scientific  << " in " << itr << " iterations.");
				return lambda; // We are done!
			}

			// Sweep Left -> Right
			for (size_t corePosition = 0; corePosition < d; ++corePosition) {
				Tensor  rhs1,rhs2, op;
				Tensor  sol,grad,xright,xleft,xx_tmp,xHx_tmp,rHx_tmp,rHr_tmp,rr_tmp,rx_tmp;
				value_t alpha,beta, xHx,rHr,rHx,xx,rr,rx;
				const Tensor &Ai = A.get_component(corePosition);
				const Tensor &Ti = T.get_component(corePosition);
				Tensor &xi = x.component(corePosition);





				begin_time = clock();

				op(i1,i2,i3,j1,j2,j3) = leftA1Stack.back()(i1, l1, j1)*Ai(l1, i2,j2, l2) *rightA1Stack.back()(i3, l2, j3);
				rhs1(i1, i2, i3) = leftAStack.back()(i1, k1,l1, j1)*\
						Ti(k1, i2, j2, k2) *\
						Ai(l1, j2,a1, l2) *\
						xi(j1, a1, j3) *\
						rightAStack.back()(i3, k2,l2, j3);

				rhs2(i1, i2, i3) = leftxStack.back()(i1, k1, j1)*\
						Ti(k1, i2, a1, k2) *\
						xi(j1, a1, j3) *\
						rightxStack.back()(i3, k2, j3);

				xHx_tmp() = op(i1^3,j1^3) * xi(i1&0) * xi(j1&0);
				//xHx_tmp() = leftA1Stack.back()(i1, l1, j1)*Ai(l1, i2,j2, l2) *rightA1Stack.back()(i3, l2, j3) * xi(i1,i2,i3) * xi(j1,j2,j3);
				xHx = xHx_tmp[0];
				xx_tmp() = xi(i1&0) * xi(i1&0);
				xx = xx_tmp[0];
				lambda = xHx/xx;
				XERUS_LOG(info,"corePosition = " << corePosition << ": " << lambda+8.80146457125193);
				stack_time += (value_t) (clock() - begin_time) / CLOCKS_PER_SEC;

				begin_time = clock();
		  	grad = rhs1 - lambda * rhs2;



		  	rHx_tmp() = op(i1^3,j1^3) * grad(i1&0) * xi(j1&0);
		  	rHx = rHx_tmp[0];
		  	rHr_tmp() = op(i1^3,j1^3) * grad(i1&0) * grad(j1&0);
				rHr = rHr_tmp[0];
				rx_tmp() = xi(i1&0) * grad(i1&0);
				rx = rx_tmp[0];
				rr_tmp() = grad(i1&0) * grad(i1&0);
				rr = rr_tmp[0];
				alpha = get_stepsize(xHx,rHr,rHx,xx,rr,rx);
		//  	alpha = 2;
		  	solving_time += (value_t) (clock() - begin_time) / CLOCKS_PER_SEC;


				x.set_component(corePosition, xi - alpha*grad);


				begin_time = clock();
				if (corePosition+1 < d) {
					x.move_core(corePosition+1, true);
					push_left_stack(corePosition);
					rightAStack.pop_back();
					rightA1Stack.pop_back();
					rightxStack.pop_back();
				}
				stack_time += (value_t) (clock() - begin_time) / CLOCKS_PER_SEC;
				//XERUS_LOG(info, "After move " << x.ranks());

			}
			// Sweep Right -> Left : only move core and update stacks
			begin_time = clock();
			x.move_core(0, true);
			for (size_t corePosition = d-1; corePosition > 0; --corePosition) {
				push_right_stack(corePosition);
				leftAStack.pop_back();
				leftA1Stack.pop_back();
				leftxStack.pop_back();
			}
			stack_time += (value_t) (clock() - begin_time) / CLOCKS_PER_SEC;


			XERUS_LOG(simpleASD, "Iteration: " << itr  <<  " Eigenvalue " << std::setprecision(10) <<  lambda +8.80146457125193 <<  " EVerr " << lambda + 8.80146457125193 +76.25663396);// << " Residual = " << calc_residual_norm());
			XERUS_LOG(simpleASD, "Iteration: " << itr  <<  " Eigenvalue " << std::setprecision(10) <<  lambda);// << " Residual = " << calc_residual_norm());
			XERUS_LOG(simpleASD,"x = " << x.ranks());
			outfile.open(out_name,std::ios::app);
			outfile <<  (value_t) (clock()-global_time)/ CLOCKS_PER_SEC << "," <<std::setprecision(12) << lambda +8.80146457125193  <<"," << stack_time <<"," << solving_time  <<  std::endl;
			outfile.close();
			result.emplace_back(lambda +8.80146457125193);
			XERUS_LOG(simpleASD,std::setprecision(10) <<result);




		}
		return lambda;
	}

	double get_stepsize(double xHx, double rHr, double rHx, double xFx, double rFr, double rFx){
		double a = rFr*rHx-rHr*rFx;
		double b = rHr*xFx-rFr*xHx;
		double c = rFx*xHx-rHx*xFx;

		double disc = b*b-4*a*c;
		double alpha1 = (-b + std::sqrt(disc))/(2*a);
		double alpha2 = (-b - std::sqrt(disc))/(2*a);
		return alpha1;
	}

};

double simpleMALS(const TTOperator& _A,const TTOperator& _T, TTTensor& _x, double _eps, size_t _maxRank)  {
	InternalSolver solver(_A, _T, _x, _eps, _maxRank);
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
