#include <xerus.h>
#include <time.h>
#include <iostream>
#include <fstream>
#include "ALSres.cpp"

#include <vector>

#include <ctime>
#include <queue>

using namespace xerus;
using xerus::misc::operator<<;

class InternalSolver;
class InternalSolver2;
double simpleALS(const TTOperator& _A, TTTensor& _x);
double simpleMALS(const TTOperator& _A, TTTensor& _x, double _eps, size_t _maxRank);


/*
 * Main!!
 */
int main() {
    

    
	XERUS_LOG(simpleALS,"Begin Tests for ordering ...");
	XERUS_LOG(simpleALS,"---------------------------------------------------------------");


	size_t d = 26; // 16 electron, 8 electron pairs
	size_t nob = 13;
	double eps =0;// 10e-15;
	size_t start_rank = 240;
	size_t max_rank = 260;
	size_t wsize = 5;
	std::string name ="hamiltonian_CH2_" + std::to_string(d) +"_full_2.ttoperator";//"henon_heiles_10_28_011.ttoperator";// 
    
    xerus::TTTensor ttx;
    std::ifstream readttx("CH2HF.tttensor");
    misc::stream_reader(readttx,ttx,xerus::misc::FileFormat::BINARY);
    /*
    ttx+=1e-4*TTTensor::random(std::vector<size_t>(d,2),2);
    ttx/=frob_norm(ttx);
    */
	xerus::TTTensor phi = xerus::TTTensor::random(std::vector<size_t>(d,2),std::vector<size_t>(d - 1,start_rank));
	xerus::TTOperator op;
	std::ifstream read(name.c_str());
	misc::stream_reader(read,op,xerus::misc::FileFormat::BINARY);
	read.close();
//	double lambda = simpleMALS(op, phi, eps, max_rank);
    double lambda = simpleALS(op, phi);
	phi.round(10e-14);



	XERUS_LOG(info, "The ranks of phi are " << phi.ranks() );
	XERUS_LOG(info, "Size of Solution " << phi.datasize() );

	XERUS_LOG(info, "Lambda =  " << std::setprecision(16) << lambda -28.1930439210	);
	XERUS_LOG(info, "Lambda Error =  " << std::setprecision(16) << std::abs(lambda -28.1930439210	+38.979392539208));

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
public:
	size_t maxIterations;

	InternalSolver(const TTOperator& _A, TTTensor& _x)
		: d(_x.degree()), x(_x), A(_A), maxIterations(200), lambda(1.0)
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
            TTTensor ALSres=TTTensor::random(x.dimensions,40);
		
                
            getRes(A,x,lambda,ALSres);
            XERUS_LOG(simpleALS, "Iteration: " << itr  <<  " EVerr " << lambda - 28.1930439210 +38.979392539208 << " Residual = " << frob_norm(ALSres) );
            //}
			/*residuals_ev.push_back(lambda);
			if (itr > 1 and std::abs(residuals_ev[residuals_ev.size()-5] - residuals_ev.back()) < 0.00005) {
				XERUS_LOG(simpleALS, "The residuum is " << calc_residual_norm());
				return lambda; // We are done!
			}*/

			// Sweep Left -> Right
			for (size_t corePosition = 0; corePosition < d; ++corePosition) {
				Tensor  rhs;
                TensorNetwork op;
                XERUS_LOG(simpleALS, "core: " << corePosition );
				const Tensor &Ai = A.get_component(corePosition);
                
            
                op(i1, i2, i3, j1, j2, j3) = leftAStack.back()(i1, k1, j1)*Ai(k1, i2, j2, k2)*rightAStack.back()(i3, k2, j3);

				//lambda = xerus::get_smallest_eigenvalue(sol, op);

                lambda = xerus::get_smallest_eigenpair_iterative(x.component(corePosition),op, false, 50000, 1e-6);
               
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

double simpleALS(const TTOperator& _A, TTTensor& _x)  {
	InternalSolver solver(_A, _x);
	return solver.solve();
}

class InternalSolver2 {
	const size_t d;
	double lambda;
	double eps;
	size_t maxRank;
	std::vector<Tensor> leftAStack;
	std::vector<Tensor> rightAStack;

	TTTensor& x;
	const TTOperator& A;
public:
	size_t maxIterations;

	InternalSolver2(const TTOperator& _A, TTTensor& _x, double _eps, size_t _maxRank)
		: d(_x.degree()), x(_x), A(_A), maxIterations(200), lambda(1.0), eps(_eps), maxRank(_maxRank)
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
        size_t roundrank=10;
        std::ofstream myfile;
        myfile.open("testMALS.dat");
        size_t currit=0;
        //std::vector<size_t> maxranks={100,100,100,100,100,100,100,100,100};
        clock_t begin_time =clock();
        time_t begin_time2 =time(NULL);
		// Build right stack
		x.move_core(0, true);
		for (size_t pos = d-1; pos > 1; --pos) {
			push_right_stack(pos);
		}

		Index i1, i2, i3, i4, j1 , j2, j3, j4, k1, k2, k3;
		Index a1, a2, a3, a4, a5, r1, r2;
		std::vector<double> residuals_ev(10, 1000.0);
		std::vector<double> residuals(10, 1000.0);
		XERUS_LOG(info,"A = " << A.ranks());

		for (size_t itr = 0; itr < maxIterations; ++itr) {
            if((itr>5)&&(itr%8==0)){
                roundrank+=10;
            }
            /*
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
            */
			// Sweep Left -> Right
			for (size_t corePosition = 0; corePosition < d-1; ++corePosition) {
				Tensor  rhs, U, S, Vt;
				TensorNetwork op;
				//XERUS_LOG(simpleMALS, "Iteration: " << itr  << " core: " << corePosition  << " Eigenvalue " << std::setprecision(16) <<  lambda);

				const Tensor &Ai = A.get_component(corePosition);
				const Tensor &Ai1 = A.get_component(corePosition+1);

				Tensor &xi = x.component(corePosition);
				Tensor &xi1 = x.component(corePosition+1);


				auto x_rank = xi.dimensions[2];
                


				//XERUS_LOG(info, "Operator Size = (" << (leftAStack.back()).dimensions[0] << "x" << Ai.dimensions[1] << "x" << Ai1.dimensions[1] << "x" << rightAStack.back().dimensions[0] << ")x("<< leftAStack.back().dimensions[2] << "x" << Ai.dimensions[2] << "x" << Ai1.dimensions[2] << "x" << rightAStack.back().dimensions[2] <<")");


				Tensor sol, xright;
				sol(a1,a2,a4,a5) = xi(a1,a2,a3)*xi1(a3,a4,a5);
                
				//lambda = xerus::get_smallest_eigenvalue(sol, op);
				op(i1, i2, i3, i4, j1, j2, j3, j4) = leftAStack.back()(i1, k1, j1)*Ai(k1, i2, j2, k2) * Ai1(k2,i3,j3,k3)*rightAStack.back()(i4, k3, j4);
		  	lambda = xerus::get_smallest_eigenpair_iterative(sol,op, false, 10000, 10e-6);
		  	//XERUS_LOG(info,sol);
       
                    //if(roundrank<maxranks[corePosition]){
                        (U(i1,i2,r1), S(r1,r2), Vt(r2,j1,j2)) = SVD(sol(i1,i2,j1,j2),1000,eps);
                    //}else{
                    //    (U(i1,i2,r1), S(r1,r2), Vt(r2,j1,j2)) = SVD(sol(i1,i2,j1,j2),maxranks[corePosition],eps);
                    //}
                


				xright(r1,j1,j2) = S(r1,r2)*Vt(r2,j1,j2);
				auto x_rank2 = U.dimensions[2];
				//ading random kicks

				auto xleft_kicked = xerus::Tensor({U.dimensions[0],U.dimensions[1],U.dimensions[2] + 1});
				auto random_kick = xerus::Tensor::random({U.dimensions[0],U.dimensions[1],1});
				xleft_kicked.offset_add(U,{0,0,0});
				xleft_kicked.offset_add(random_kick,{0,0,U.dimensions[2]});
				auto xright_kicked = xerus::Tensor({xright.dimensions[0] + 1,xright.dimensions[1],xright.dimensions[2]});
				xright_kicked.offset_add(xright,{0,0,0});

				//XERUS_LOG(info, "U " << U.dimensions << " Vt " << xright.dimensions);
				x.set_component(corePosition, xleft_kicked);
				x.set_component(corePosition+1, xright_kicked);\
				//XERUS_LOG(info, "After kick " << x.ranks());

//				if (x_rank < x_rank2){
//					leftAStack.clear();
//					leftAStack.emplace_back(Tensor::ones({1,1,1}));
//					for (size_t pos = 0; pos < corePosition; ++pos ){
//						push_left_stack(pos);
//					}
//				}



				if (corePosition+2 < d) {
					x.move_core(corePosition+1, true);
					push_left_stack(corePosition);
					rightAStack.pop_back();
				}
				//XERUS_LOG(info, "After move " << x.ranks());

			}
			TTTensor ALSres=TTTensor::random(x.dimensions,200);
			//if(currit%3==1){
                
                getRes(A,x,lambda,ALSres);
            //}
			myfile<<std::setprecision(4)<<frob_norm(ALSres)<<std::setw(12)<<"\t"<<std::setprecision(12)<<lambda/*- 28.1930439210 +38.979392539208*/<<std::setw(12)<<"\t"<<currit<<std::setw(12)<<"\t"<<(clock()-begin_time)*1e-7/35<<std::setw(12)<<"\t"<<(time(NULL)-begin_time2)<<std::endl;
            currit++;
            
            
			XERUS_LOG(simpleALS, "Iteration: " << currit  <<  " EVerr " << lambda/* - 28.1930439210 +38.979392539208 */<< " Residual = " << frob_norm(ALSres) );
            XERUS_LOG(info,"x = " << x.ranks());

			// Sweep Right -> Left : only move core and update stacks
			x.move_core(0, true);
			for (size_t corePosition = d-1; corePosition > 1; --corePosition) {
				push_right_stack(corePosition);
				leftAStack.pop_back();
			}

		}
		return lambda;
	}
};

double simpleMALS(const TTOperator& _A, TTTensor& _x, double _eps, size_t _maxRank)  {
	InternalSolver2 solver(_A, _x, _eps, _maxRank);
	return solver.solve();
}

