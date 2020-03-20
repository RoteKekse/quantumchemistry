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
class InternalSolver2;
double simpleALS(const TTOperator& _A, TTTensor& _x, const TTOperator& _P, size_t _nosw);
double simpleMALS(const TTOperator& _A, TTTensor& _x, double _eps, size_t _maxRank, size_t _nosw);
void write_to_disc(std::string name, TTTensor &x);
void write_to_disc(std::string name, Tensor &x);
void write_to_disc(std::string name, TTOperator &op);
void read_from_disc(std::string name, TTOperator &op);
void read_from_disc(std::string name, TTTensor &x);
void project(TTTensor &phi, size_t p, size_t d);
TTOperator particleNumberOperator(size_t k, size_t d);
TTOperator particleNumberOperator(size_t d);
/*
 * Main!!
 */
int main() {
	XERUS_LOG(simpleALS,"Begin Tests for ordering ...");
	XERUS_LOG(simpleALS,"---------------------------------------------------------------");
	Index ii,jj;
	//Set Parameters
	size_t d = 48;
	size_t part = 48 ;
	size_t p = 8;
	size_t nob = part / 2;
	double eps = 10e-6;
	size_t start_rank = 1;
	size_t max_rank = 20;
	size_t number_of_sweeps = 20;

	//Load Intial Value
	xerus::TTTensor phi(std::vector<size_t>(part,2));
	read_from_disc("../data/eigenvector_H2O_48_3_-23.647510_benchmark.tttensor",phi);
	//read_from_disc("../data/eigenvector_H2O_48_5_-23.700175_benchmark.tttensor",phi);
	//phi = TTTensor::random(std::vector<size_t>(part,2),std::vector<size_t>(part-1,start_rank));
	project(phi,p,d);
	phi.round(10);
	phi.round(1e-6);
	phi/=phi.frob_norm();
	XERUS_LOG(info,"Ranks Phi = \n" << phi.ranks());

	//Load Hamiltonian
	xerus::TTOperator op;
	read_from_disc("../data/hamiltonian_H2O_" + std::to_string(d)  +"_full_benchmark.ttoperator",op);

	//Calculate initial energy
	Tensor hf;
	hf() = op(ii/2,jj/2)*phi(ii&0)*phi(jj&0);
	XERUS_LOG(info,"Initial Energy " << hf[0]-52.4190597253);

	//Calculate particle number
	auto P = particleNumberOperator(d);
	Tensor pn;
	pn() = P(ii/2,jj/2)*phi(ii&0)*phi(jj&0);
	XERUS_LOG(info,"Initial Particle Number " << std::setprecision(16)<< pn[0]);




	//Perfrom ALS/DMRG
	double lambda = simpleMALS(op, phi, eps, max_rank,number_of_sweeps);
//	double lambda = simpleALS(op, phi,P, number_of_sweeps);
	phi.round(10e-14);


	XERUS_LOG(info, "The ranks of phi are " << phi.ranks() );
	XERUS_LOG(info, "Size of Solution " << phi.datasize() );

	XERUS_LOG(info, "Final Energy =  " << std::setprecision(16) << lambda 	-52.4190597253);
	phi /= phi.frob_norm();
	pn() = P(ii/2,jj/2)*phi(ii&0)*phi(jj&0);
	XERUS_LOG(info,"Final Particle Number " << std::setprecision(16)<< pn[0]);

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
	std::vector<Tensor> leftAStack;
	std::vector<Tensor> rightAStack;

	TTTensor& x;
	const TTOperator& A;
	const TTOperator& P;
public:
	size_t maxIterations;

	InternalSolver(const TTOperator& _A, TTTensor& _x, const TTOperator& _P, size_t _nosw)
		: d(_x.order()), x(_x), A(_A), P(_P), maxIterations(_nosw), lambda(1.0)
	{
		leftAStack.emplace_back(Tensor::ones({1,1,1}));
		rightAStack.emplace_back(Tensor::ones({1,1,1}));
	}


	void push_left_stack(const size_t _position) {
		Index i1, i2, i3, j1 , j2, j3, k1, k2;
		const Tensor &xi = x.get_component(_position);
		const Tensor &Ai = A.get_component(_position);

		Tensor tmpA;
		tmpA(i1, i2, i3) = leftAStack.back()(j1, j2, j3)
				*xi(j1, k1, i1)*Ai(j2, k1, k2, i2)*xi(j3, k2, i3);
		leftAStack.emplace_back(std::move(tmpA));
	}


	void push_right_stack(const size_t _position) {
		Index i1, i2, i3, j1 , j2, j3, k1, k2;
		const Tensor &xi = x.get_component(_position);
		const Tensor &Ai = A.get_component(_position);

		Tensor tmpA;
		tmpA(i1, i2, i3) = xi(i1, k1, j1)*Ai(i2, k1, k2, j2)*xi(i3, k2, j3)
				*rightAStack.back()(j1, j2, j3);
		rightAStack.emplace_back(std::move(tmpA));
	}

	double calc_residual_norm() {
		Index i,j;
		return frob_norm(A(i/2, j/2)*x(j&0) - lambda*x(i&0));
	}


	double solve() {
		// Build right stack
		x.move_core(0, true);
		for (size_t pos = d-1; pos > 0; --pos) {
			push_right_stack(pos);
		}




		Index i1, i2, i3, i4, j1 , j2, j3, j4, k1, k2, k3;
		std::vector<double> residuals_ev(10, 1000.0);
		for (size_t itr = 0; itr < maxIterations; ++itr) {
			// Calculate residual and check end condition
			XERUS_LOG(simpleALS, "Iteration: " << itr << " Eigenvalue " << std::setprecision(16) <<  lambda);
			residuals_ev.push_back(lambda);
			if (itr > 1 and std::abs(residuals_ev[residuals_ev.size()-5] - residuals_ev.back()) < 0.00005) {
				XERUS_LOG(simpleALS, "The residuum is " << calc_residual_norm());
				return lambda; // We are done!
			}

			// Sweep Left -> Right
			for (size_t corePosition = 0; corePosition < d; ++corePosition) {
				Tensor op, rhs, pn,pn2,en,tmp;
				TTTensor xtmp;

				const Tensor &Ai = A.get_component(corePosition);

				op(i1, i2, i3, j1, j2, j3) = leftAStack.back()(i1, k1, j1)*Ai(k1, i2, j2, k2)*rightAStack.back()(i3, k2, j3);
				if (corePosition == 2)
					write_to_disc("../data/test_hamiltoninan_instability_pos1.tttensor",x);

				//lambda = xerus::get_smallest_eigenvalue(sol, op);
				lambda = xerus::get_eigenpair_iterative(x.component(corePosition),op, true,false, 50000, 1e-14);
				//XERUS_LOG(info,sol);
				x/=x.frob_norm();

				pn() = P(i1/2,j1/2)*x(i1&0)*x(j1&0);
				if (corePosition == 2)
					write_to_disc("../data/test_hamiltoninan_instability_pos2.tttensor",x);

				XERUS_LOG(info,"Particle Number step " << corePosition << " " << std::setprecision(16)<< pn[0] << " Eigenvalue = " << lambda-52.4190597253);

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

double simpleALS(const TTOperator& _A, TTTensor& _x, const TTOperator& _P, size_t _nosw)  {
	InternalSolver solver(_A, _x,_P, _nosw);
	return solver.solve();
}

class InternalSolver2 {
	const size_t d;
	double lambda;
	double eps;
	size_t maxRank;
	std::vector<Tensor> leftAStack;
	std::vector<Tensor> rightAStack;
	value_t nuc = -52.4190597253,ref = -76.25663396;
	TTTensor& x;
	const TTOperator& A;
	TTOperator P;
public:
	size_t maxIterations;

	InternalSolver2(const TTOperator& _A, TTTensor& _x,  double _eps, size_t _maxRank, size_t _nosw)
		: d(_x.order()), x(_x), A(_A), maxIterations(_nosw), lambda(1.0), eps(_eps), maxRank(_maxRank)
	{
		leftAStack.emplace_back(Tensor::ones({1,1,1}));
		rightAStack.emplace_back(Tensor::ones({1,1,1}));
		P = particleNumberOperator(d);
	}


	void push_left_stack(const size_t _position) {
		Index i1, i2, i3, j1 , j2, j3, k1, k2;
		const Tensor &xi = x.get_component(_position);
		const Tensor &Ai = A.get_component(_position);


		Tensor tmpA;
		tmpA(i1, i2, i3) = leftAStack.back()(j1, j2, j3)
				*xi(j1, k1, i1)*Ai(j2, k1, k2, i2)*xi(j3, k2, i3);
		leftAStack.emplace_back(std::move(tmpA));
	}


	void push_right_stack(const size_t _position) {
		Index i1, i2, i3, j1 , j2, j3, k1, k2;
		const Tensor &xi = x.get_component(_position);
		const Tensor &Ai = A.get_component(_position);

		Tensor tmpA;
		tmpA(i1, i2, i3) = xi(i1, k1, j1)*Ai(i2, k1, k2, j2)*xi(i3, k2, j3)
				*rightAStack.back()(j1, j2, j3);
		rightAStack.emplace_back(std::move(tmpA));
	}

	double calc_residual_norm() { // TODO improve this by (A-lamdaI)
		Index ii,jj,kk,ll,mm,nn,oo,i1,i2,i3,i4;
		auto ones = Tensor::ones({1,1,1});
		xerus::Tensor tmp = ones;
		XERUS_LOG(info,"lambda = " << lambda - 28.1930439210);

		for (size_t i = 0; i < d; i++){
			auto Ai = A.get_component(i);
			auto xi = x.get_component(i);
			tmp(i1,i2,i3) = tmp(ii,jj,kk) * xi(ii,ll,i1) * Ai(jj,ll,mm,i2) * xi(kk,mm,i3);
		}
		tmp() = tmp(ii,jj,kk) * ones(ii,jj,kk);
		XERUS_LOG(info,"xAx = " << tmp);

		auto ones2 = Tensor::ones({1,1,1,1});
		xerus::Tensor tmp2 = ones2;
		for (size_t i = 0; i < d; i++){
			auto Ai = A.get_component(i);
			auto xi = x.get_component(i);
			//XERUS_LOG(info,i);
			//XERUS_LOG(info,tmp2.dimensions);
			tmp2(i1,i2,i3,i4) = tmp2(ii,jj,kk,ll) * xi(ii,mm,i1) * Ai(jj,mm,nn,i2) * Ai(kk,nn,oo,i3) * xi(ll,oo,i4);
		}
		tmp2() = tmp2(ii,jj,kk,ll) * ones2(ii,jj,kk,ll);

	//	XERUS_LOG(info,"xAAx = " << tmp2);

		xerus::TTTensor tmp3;
//		tmp3(ii&0) = A(ii/2,jj/2) * x(jj&0);
//		XERUS_LOG(info,tmp3.ranks());
		return std::sqrt(std::abs(tmp2[0]-lambda*lambda));
	}


	double solve() {
		// Build right stack
		x.move_core(0, true);
		for (size_t pos = d-1; pos > 1; --pos) {
			push_right_stack(pos);
		}

	  std::ofstream outfile;
		std::string out_name = "/homes/numerik/goette/Documents/jupyter_examples/Paper/Preconditioning/data/DMRG_rank_" + std::to_string(maxRank) +"_cpu_benchmark.csv";
		outfile.open(out_name);
		outfile.close();
	  clock_t begin_time,global_time = clock();
	  value_t stack_time = 0,solving_time = 0;

		Index i1, i2, i3, i4, j1 , j2, j3, j4, k1, k2, k3;
		Index a1, a2, a3, a4, a5, r1, r2;
		std::vector<double> residuals_ev(10, 1000.0);
		std::vector<double> residuals(10, 1000.0);
		XERUS_LOG(info,"A = " << A.ranks());
		std::vector<value_t> result;

		for (size_t itr = 0; itr < maxIterations; ++itr) {
//			if (itr % 5 == 0 and itr != 0)
//				maxRank = 250 > maxRank + 10 ? maxRank + 10 : 250;
			stack_time = 0,solving_time = 0;


			// Calculate residual and check end condition
			residuals_ev.push_back(lambda);
			//residuals.push_back(calc_residual_norm());
			if (itr > 1 and std::abs(residuals_ev[residuals_ev.size()-10] - residuals_ev.back()) < eps) {
				XERUS_LOG(info, residuals_ev[residuals_ev.size()-10]);
				XERUS_LOG(info, residuals_ev.back());
				XERUS_LOG(info, eps);
				XERUS_LOG(simpleMALS, "Done! Residual decreased to residual "  << std::scientific  << " in " << itr << " iterations.");
				return lambda; // We are done!
			}

			// Sweep Left -> Right
			for (size_t corePosition = 0; corePosition < d-1; ++corePosition) {
				Tensor  rhs,pn;
				TensorNetwork op;
				//Tensor op;
				//XERUS_LOG(simpleMALS, "Iteration: " << itr  << " core: " << corePosition  << " Eigenvalue " << std::setprecision(16) <<  lambda);

				const Tensor &Ai = A.get_component(corePosition);
				const Tensor &Ai1 = A.get_component(corePosition+1);

				Tensor &xi = x.component(corePosition);
				Tensor &xi1 = x.component(corePosition+1);


				auto x_rank = xi.dimensions[2];



				//XERUS_LOG(info, "Operator Size = (" << (leftAStack.back()).dimensions[0] << "x" << Ai.dimensions[1] << "x" << Ai1.dimensions[1] << "x" << rightAStack.back().dimensions[0] << ")x("<< leftAStack.back().dimensions[2] << "x" << Ai.dimensions[2] << "x" << Ai1.dimensions[2] << "x" << rightAStack.back().dimensions[2] <<")");


				Tensor sol, xright,xleft;
				sol(a1,a2,a4,a5) = xi(a1,a2,a3)*xi1(a3,a4,a5);

				//lambda = xerus::get_smallest_eigenvalue(sol, op);
				begin_time = clock();
				op(i1, i2, i3, i4, j1, j2, j3, j4) = leftAStack.back()(i1, k1, j1)*Ai(k1, i2, j2, k2) * Ai1(k2,i3,j3,k3)*rightAStack.back()(i4, k3, j4);
				stack_time += (value_t) (clock() - begin_time) / CLOCKS_PER_SEC;
//				XERUS_LOG(info,op.dimensions);

				begin_time = clock();
				lambda = xerus::get_eigenpair_iterative(sol,op, true,false, 100000, eps);

				auto xnew = split1(sol,maxRank,1e-6);
				solving_time += (value_t) (clock() - begin_time) / CLOCKS_PER_SEC;
				//auto xnew = split2(op,sol,maxRank,eps);


				xright = xnew.second;
				xleft  = xnew.first;

//				auto x_rank2 = xleft.dimensions[2];
//				//adding random kicks
//				auto xleft_kicked = xerus::Tensor({xleft.dimensions[0],xleft.dimensions[1],xleft.dimensions[2] + 1});
//				auto random_kick = xerus::Tensor::random({xleft.dimensions[0],xleft.dimensions[1],1});
//				xleft_kicked.offset_add(xleft,{0,0,0});
//				xleft_kicked.offset_add(random_kick,{0,0,xleft.dimensions[2]});
//				auto xright_kicked = xerus::Tensor({xright.dimensions[0] + 1,xright.dimensions[1],xright.dimensions[2]});
//				xright_kicked.offset_add(xright,{0,0,0});

				//XERUS_LOG(info, "U " << U.dimensions << " Vt " << xright.dimensions);
				if (corePosition == 24)
					write_to_disc("../data/test_hamiltoninan_instability_pos23.tttensor",x);
				x.set_component(corePosition, xleft);
				x.set_component(corePosition+1, xright);

				x/=x.frob_norm();
				if (corePosition == 24)
					write_to_disc("../data/test_hamiltoninan_instability_pos24.tttensor",x);
				pn() = P(i1/2,j1/2)*x(i1&0)*x(j1&0);
				XERUS_LOG(info,"Particle Number step " << corePosition << " " << std::setprecision(16)<< pn[0] << " Eigenvalue = " << lambda-52.4190597253);


				//XERUS_LOG(info, "After kick " << x.ranks());

//				if (x_rank < x_rank2){
//					leftAStack.clear();
//					leftAStack.emplace_back(Tensor::ones({1,1,1}));
//					for (size_t pos = 0; pos < corePosition; ++pos ){
//						push_left_stack(pos);
//					}
//				}


				begin_time = clock();
				if (corePosition+2 < d) {
					x.move_core(corePosition+1, true);
					push_left_stack(corePosition);
					rightAStack.pop_back();
				}
				stack_time += (value_t) (clock() - begin_time) / CLOCKS_PER_SEC;
				//XERUS_LOG(info, "After move " << x.ranks());

			}
			// Sweep Right -> Left : only move core and update stacks
			begin_time = clock();
//			project(x,8,d);
//			x.round(1e-8);
//			x.round(maxRank);
//			x/= x.frob_norm();
			x.move_core(0, true);

			for (size_t corePosition = d-1; corePosition > 1; --corePosition) {
				push_right_stack(corePosition);
				leftAStack.pop_back();
			}
			stack_time += (value_t) (clock() - begin_time) / CLOCKS_PER_SEC;

			if (false/*itr % 5 == 0 and itr != 0*/){

				TTTensor res = TTTensor::random(std::vector<size_t>(d, 2), {maxRank});
				getRes(A,x,lambda,res);
				XERUS_LOG(simpleALS, "Iteration: " << itr  << " Res " << res.frob_norm() <<   " Eigenvalue " << std::setprecision(10) <<  lambda +nuc<<  " EVerr " << lambda -52.4190597253 +76.25663396);// << " Residual = " << calc_residual_norm());
				XERUS_LOG(info,"x = " << x.ranks());
			} else {
				XERUS_LOG(simpleALS, "Iteration: " << itr  <<  " Eigenvalue " << std::setprecision(10) <<  lambda +nuc <<  " EVerr " << lambda +nuc - ref);// << " Residual = " << calc_residual_norm());
				XERUS_LOG(simpleALS, "Iteration: " << itr  <<  " Eigenvalue " << std::setprecision(10) <<  lambda);// << " Residual = " << calc_residual_norm());
				XERUS_LOG(info,"x = " << x.ranks());
				outfile.open(out_name,std::ios::app);
				outfile <<  (value_t) (clock()-global_time)/ CLOCKS_PER_SEC << "," <<std::setprecision(12) << lambda +nuc <<"," << stack_time <<"," << solving_time  <<  std::endl;
				outfile.close();
				result.emplace_back(lambda +nuc);
				XERUS_LOG(info,std::setprecision(10) <<result);

			}


		}
		return lambda;
	}

	std::pair<Tensor,Tensor> split1(Tensor& sol, size_t maxRank, value_t eps){
		Tensor  U, S, Vt,xright;
		Index i1,i2,r1,r2,j1,j2;
		(U(i1,i2,r1), S(r1,r2), Vt(r2,j1,j2)) = SVD(sol(i1,i2,j1,j2),maxRank,eps);
		xright(r1,j1,j2) = S(r1,r2)*Vt(r2,j1,j2);
		return std::pair<Tensor,Tensor>(U,xright);
	}

	std::pair<Tensor,Tensor> split2(const TensorNetwork op, Tensor& sol, size_t maxRank, value_t eps){
		Tensor  U, S, Vt,xright;
		Index i1,i2,i3,i4,r1,r2,j1,j2,j3,j4;
		(U(i1,i2,r1), S(r1,r2), Vt(r2,j1,j2)) = SVD(sol(i1,i2,j1,j2),maxRank,eps);
		xright(r1,j1,j2) = S(r1,r2)*Vt(r2,j1,j2);

		std::pair<Tensor,Tensor> x = std::pair<Tensor,Tensor>(U,xright);
		Tensor oploc,rhs;
		for (size_t i = 0; i < 10; ++i){
			oploc(i1,i2,r1,j1,j2,r2)  = op(i1,i2,i3,i4,j1,j2,j3,j4) * x.second(r2,j3,j4) * x.second(r1,i3,i4);

			rhs(i1,i2,r1) = op(i1,i2,i3,i4,j1,j2,j3,j4) * sol(j1,j2,j3,j4) * x.second(r1,i3,i4);

			xerus::solve(x.first,oploc,rhs,0);

			oploc(r1,i3,i4,r2,j3,j4)  = op(i1,i2,i3,i4,j1,j2,j3,j4) * x.first(j1,j2,r2) * x.first(i1,i2,r1);
			rhs(r1,i3,i4) = op(i1,i2,i3,i4,j1,j2,j3,j4) * sol(j1,j2,j3,j4) * x.first(i1,i2,r1);

			xerus::solve(x.second,oploc,rhs,0);


		}


		return x;
	}

};

double simpleMALS(const TTOperator& _A, TTTensor& _x, double _eps, size_t _maxRank, size_t _nosw)  {
	InternalSolver2 solver(_A, _x, _eps, _maxRank,_nosw);
	return solver.solve();
}


void write_to_disc(std::string name, Tensor &x){
	std::ofstream write(name.c_str() );
	xerus::misc::stream_writer(write,x,xerus::misc::FileFormat::BINARY);
	write.close();
}

void write_to_disc(std::string name, TTTensor &x){
	std::ofstream write(name.c_str() );
	xerus::misc::stream_writer(write,x,xerus::misc::FileFormat::BINARY);
	write.close();
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

void project(TTTensor &phi, size_t p, size_t d){
	Index i1,i2,j1,j2,k1,k2;
	for (size_t k = 0; k <= d; ++k){

		if (p != k){
			auto PNk = particleNumberOperator(k,d);
			PNk.move_core(0);
			phi(i1&0) = PNk(i1/2,k1/2) * phi (k1&0);
			value_t f = (value_t) p - (value_t) k;
			phi /=  f;
			//phi.round(1e-12);
			phi /= phi.frob_norm();

		}
	}
	phi /= phi.frob_norm();
}



TTOperator particleNumberOperator(size_t k, size_t d){
	TTOperator op(std::vector<size_t>(2*d,2));
	Tensor id = Tensor::identity({2,2});
	id.reinterpret_dimensions({1,2,2,1});
	auto n = id;
	n[{0,0,0,0}] = 0;

	value_t kk = (value_t) k;
	auto kkk = kk/(value_t) d * id;
	Tensor tmp({1,2,2,2});
	tmp.offset_add(id,{0,0,0,0});
	tmp.offset_add(n-kkk,{0,0,0,1});
	op.set_component(0,tmp);

	for (size_t i = 1; i < d-1; ++i){
		tmp = Tensor({2,2,2,2});
		tmp.offset_add(id,{0,0,0,0});
		tmp.offset_add(id,{1,0,0,1});
		tmp.offset_add(n-kkk,{0,0,0,1});
		op.set_component(i,tmp);
	}
  tmp = Tensor({2,2,2,1});
	tmp.offset_add(n-kkk,{0,0,0,0});
	tmp.offset_add(id,{1,0,0,0});
	op.set_component(d-1,tmp);


	return op;
}

TTOperator particleNumberOperator(size_t d){
	TTOperator op(std::vector<size_t>(2*d,2));
	Tensor id = Tensor::identity({2,2});
	id.reinterpret_dimensions({1,2,2,1});
	auto n = id;
	n[{0,0,0,0}] = 0;
	Tensor tmp({1,2,2,2});
	tmp.offset_add(id,{0,0,0,0});
	tmp.offset_add(n,{0,0,0,1});
	op.set_component(0,tmp);
	for (size_t i = 1; i < d-1; ++i){
		tmp = Tensor({2,2,2,2});
		tmp.offset_add(id,{0,0,0,0});
		tmp.offset_add(id,{1,0,0,1});
		tmp.offset_add(n,{0,0,0,1});
		op.set_component(i,tmp);
	}
  tmp = Tensor({2,2,2,1});
	tmp.offset_add(n,{0,0,0,0});
	tmp.offset_add(id,{1,0,0,0});
	op.set_component(d-1,tmp);


	return op;
}
