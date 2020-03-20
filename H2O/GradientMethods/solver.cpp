#include <xerus.h>
#include <time.h>
#include <cmath>
#include <iostream>
#include <fstream>            
#include "localProblem.cpp"
#include "eigen.cpp"
//#include "ALSres.cpp"

using namespace xerus;
using xerus::misc::operator<<;


class solver{
    TTTensor& x;
    const TTOperator& A;
    const TTOperator& F;
    TTOperator FF;

    std::vector<Tensor> xloc;
    std::vector<Tensor> Axloc;
    std::vector<Tensor> FFxloc;
    std::vector<Tensor> gradloc;
    
    TTTensor ALSres;
    
    std::vector<size_t> ranks;
    
    localProblem local1;
   
    value_t res_norm = 1.0;
    size_t currit;
    size_t maxrankfactor=1;
public:
    size_t CG_minit=2;
    size_t CG_maxit=20;
    value_t CG_error=1e-6;
    
    std::vector<size_t> maxranks;
    value_t rounderror = 10e-12;
    double switchlambda;
    
    bool calcresidual;
    
    double target_lambda;
    double steplength = 0.01;
    double lambda = 0.0;
    clock_t begin_time;
    time_t begin_time2;
    
  //  double testlambda;
    
    
   solver(const TTOperator& _A, const TTOperator& _F, TTTensor& _x,double _targetlambda)
		:local1(x),  x(_x), A(_A), F(_F)
		{
			XERUS_LOG(info,"target eigenvalue: "<<_targetlambda);
			Index ii,jj,kk;
			FF = TTOperator(F.dimensions);
			FF(ii/2,jj/2) = F(ii/2,kk/2) *  F(kk/2,jj/2);

			currit=0;
			switchlambda=-1e20;
			target_lambda=_targetlambda;
			ranks=x.ranks();
			maxranks=ranks;

			x/=op_norm(x,FF);
			calcresidual=false;

			begin_time =clock();
			begin_time2 =time(NULL);


	}
	
	void tangentsubspacespacesolve(size_t maxit){
        local1.update(x);
        xloc=local1.getlocalVector(x);
       
        
        
        std::vector<std::vector<Tensor>> V;
        V.emplace_back(xloc);
        std::vector<std::vector<Tensor>> AV;
        writeData(true);
        AV.emplace_back(Axloc);
        auto H=Tensor({V.size(),V.size()});
        H[0]=lambda;
        Tensor eigen=Tensor::ones({1});
        for (size_t it=0;it<maxit;it++){
            auto v=gradloc;
            getdirection(v);
            auto tmp=frob_norm(v);
            //orthonormalize against V
			for (size_t pos=0;pos<V.size();pos++){
                
				auto h=innerprod(v,V[pos]);
                
				add(v,V[pos],-h);
				
			}
			//std::cout<<frob_norm(v)/tmp<<std::endl;
			
            multiply(v,1/frob_norm(v));
            
            V.emplace_back(v);
		//built A*V
			
            
            auto Av=local1.localProduct(v,A);
            
            AV.emplace_back(Av);
            
            H.resize_mode(0,V.size(),H.dimensions[0]);
			H.resize_mode(1,V.size(),H.dimensions[1]);
            
            for (size_t l=0;l<V.size();l++){
					
                    auto h=innerprod(AV[l],v);
                   
					H[{l,V.size()-1}]=h;
                    H[{V.size()-1,l}]=h;
                    
				
			}
			
			eigen.resize_mode(0,V.size(),V.size()-1);
            std::cout<<eigen.to_string()<<std::endl;
            targeteigen(H,eigen,1e2,1e-5, target_lambda-1);
			for (size_t pos=0;pos<V.size();pos++){
                if (pos==0){
                    xloc=V[0];
                    multiply(xloc,eigen[pos]);
                    
                }else{
                    add(xloc,V[pos],eigen[pos]);
                }
			}
			
            std::cout<<"TEST:  "<<frob_norm(eigen)<<std::endl;
          //  std::cout<<"TEST:  "<<frob_norm(xloc)<<std::endl;
            writeData(false);
            
        }
        x=local1.builtTTTensor(xloc);
        adapt_ranks();
        x/=frob_norm(x);
       
    }
    
	void tangentspacesimple(size_t maxit){
			local1.update(x);
			xloc=local1.getlocalVector(x);
			writeData(true);
			for(size_t it=0;it<maxit;it++){
					auto v=gradloc;
					getdirection(v);
					auto tmp=frob_norm(v);
					add(xloc,v,(-1));
					multiply(xloc,1/frob_norm(xloc));
					writeData(false);

			}
			x=local1.builtTTTensor(xloc);
			adapt_ranks();
			x/=frob_norm(x);

	}
	
	
	void simple(){
		Index i,j,k;
		local1.update(x);
		xloc=local1.getlocalVector(x);
		writeData(true); //updates gradloc
		auto v=gradloc;
		getdirection(v);
		steplength=-1;
		add(xloc,v,(steplength));
		x=local1.builtTTTensor(xloc);
		XERUS_LOG(info,x.ranks());
		//adapt_ranks();
		x.round(maxranks);
		XERUS_LOG(info,x.ranks());
		x/=op_norm(x,FF);
	}
    
	void getdirection(std::vector<Tensor>& direction){
		TTOperator Ilambda;
		value_t target_lambda_tmp = switchlambda<lambda ? target_lambda : lambda ;
		Ilambda= (-1) * target_lambda_tmp * FF;
		if(CG_maxit>0){
			local1.krylow(A,Ilambda,F,FF,xloc,gradloc,direction,CG_maxit,CG_error);
		}
	}

    
  void writeData(bool simple){
		Axloc=local1.localProduct(A,F);
		FFxloc=local1.localProduct(FF);

		gradloc=Axloc;
		project(gradloc,xloc, FFxloc); //TODO add FF so it is (I-Fxx^T) Ax, check if project method is correct

		res_norm=frob_norm(gradloc);
		lambda=innerprod(Axloc,xloc); // OK for PC since x is normed wrt FF
		Index i,j;
	}

	void adapt_ranks(){
		if(true)//{
				x=local1.retraction_ALS(xloc);
//		}else{
//				x=local1.retraction_ALS(xloc);
//		}
  }

};
