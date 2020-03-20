#include <xerus.h>
#include <time.h>
#include <cmath>
#include <iostream>
#include <fstream>            
#include "localProblem.cpp"
#include "eigen.cpp"
#include "ALSres.cpp"

using namespace xerus;


class solver{
    TTTensor& x;
    const TTOperator& A;

    std::vector<Tensor> xloc;
    std::vector<Tensor> Axloc;
    std::vector<Tensor> gradloc;
    
    TTTensor ALSres;
    
    std::vector<size_t> ranks;
    
    localProblem local1;
   
    value_t res_norm;
    size_t currit;
    size_t maxrankfactor=1;
public:
    size_t CG_minit=2;
    size_t CG_maxit=20;
    value_t CG_error=1e-6;
    
    std::vector<size_t> maxranks;
    value_t rounderror;
    double switchlambda;
    
    bool calcresidual;
    
    double target_lambda;
    double steplength;
    double lambda;
    clock_t begin_time;
    time_t begin_time2;
    
  //  double testlambda;
    
    
    solver(const TTOperator& _A, TTTensor& _x,double _targetlambda,std::ofstream& file) 
	:local1(x),  x(_x), A(_A)
	{ 
   /*     TTTensor CH2;
       std::ifstream readttx("CH2res6.901e-13 .tttensor");
       misc::stream_reader(readttx,CH2,xerus::misc::FileFormat::BINARY);
       localProblem sol(CH2);
       auto tmp1=sol.getlocalVector(CH2);
       auto tmp2=sol.localProduct(A);
       testlambda=innerprod(tmp2,tmp1);
        
        
        */
        
        std::cout<<" "<<std::endl;
        currit=0;
        using xerus::misc::operator<<;

		
        switchlambda=-1e20;
        target_lambda=_targetlambda;		
        XERUS_LOG(info,"target eigenvalue: "<<target_lambda);
        ranks=x.ranks();
        maxranks=ranks;
        x/=frob_norm(x);
        calcresidual=false;

        begin_time =clock();
        begin_time2 =time(NULL);
        
	}
	
	void subspacetransportsolve(size_t maxit){
        local1.update(x);
        xloc=local1.getlocalVector(x);
       
        
        
        std::vector<std::vector<Tensor>> V;
        V.emplace_back(xloc);
        
        
        Tensor eigen=Tensor::ones({1});
        for (size_t it=0;it<maxit;it++){
            writeData(true);
            auto v=gradloc;
            getdirection(v);
            
            
            for (size_t pos=0;pos<V.size();pos++){
                
				auto h=innerprod(v,V[pos]);
                
				add(v,V[pos],-h);
				
			}
			
			
            multiply(v,1/frob_norm(v));
            
            V.emplace_back(v);
            
            
            std::vector<std::vector<Tensor>> AV;
            for (size_t pos=0;pos<V.size();pos++){
                auto tmp =local1.localProduct(V[pos],A);
                AV.emplace_back(tmp);
				
			}
			Tensor H=Tensor({V.size(),V.size()});
			for (size_t i=0;i<V.size();i++){
                for (size_t j=i;j<V.size();j++){
                    auto tmp=innerprod(V[i],AV[j]);
                    H[{j,i}]=tmp;
                    H[{i,j}]=tmp;
                }
            }
            eigen.resize_mode(0,V.size(),V.size()-1);
            xerus::get_smallest_eigenpair_iterative(eigen,H, false, 10000, 10e-5);
            std::cout<<eigen.to_string()<<std::endl;
			for (size_t pos=0;pos<V.size();pos++){
                if (pos==0){
                    xloc=V[0];
                    multiply(xloc,eigen[pos]);
                    
                }else{
                    add(xloc,V[pos],eigen[pos]);
                }
			}
// 			add(xloc,V[0]);
//             multiply(xloc,.5);
			
			auto localalt=local1;
            
            x=local1.builtTTTensor(xloc);
            adapt_ranks();
            x/=frob_norm(x);
            local1.update(x);
            xloc=local1.getlocalVector(x);
            V[0]=xloc;
            
          //  std::cout<<"TEST:  "<<frob_norm(eigen)<<std::endl;
          //  std::cout<<"TEST:  "<<frob_norm(xloc)<<std::endl;

            if(it<maxit-1){
                for (size_t pos=1;pos<V.size();pos++){
                    auto tmp=local1.vector_transport(V[pos],localalt);
                    
                    for (size_t pos2=0;pos2<pos;pos2++){
                        auto h=innerprod(tmp,V[pos2]);
                        add(tmp,V[pos2],-h);
                    }
                    multiply(tmp,1/frob_norm(tmp));
                    V[pos]=tmp;
                }
            }
            
            
            
        }
        
        
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

        writeData(true);
        auto v=gradloc;
        getdirection(v);

        steplength=-1;
        add(xloc,v,(steplength));

        x=local1.builtTTTensor(xloc);

        adapt_ranks();

        
        x/=frob_norm(x);
    }
    
    void getdirection(std::vector<Tensor>& direction){
        TTOperator Op,I;
        I=TTOperator::identity(A.dimensions);
        if(switchlambda<lambda){
            Op=A-target_lambda*I;
        }else{
            Op=A-lambda*I;
        }
        if(CG_maxit>0){
        //multiply(direction,-1);
        //local1.cg_method(Op,xloc,gradloc,direction,1,1,CG_error);
        local1.krylow(Op,xloc,gradloc,direction,CG_maxit,CG_error);

        
        }
//         local1.krylowtransport(Op,xloc,gradloc,direction,CG_maxit,CG_error);
        
//             auto tmp=local1.localProduct(direction,Op);
//             project(tmp,xloc);
// 
// 
// 
//             add(tmp,gradloc,(-1));
// 
//            std::cout<<"Testiger Testboi Ax-b: "<<frob_norm(tmp)/frob_norm(gradloc)<<std::endl;


    }
    
    
    void writeData(bool simple){
        
        
        
        if (simple){
            Axloc=local1.localProduct(A);
        }else{
            Axloc=local1.localProduct(xloc,A); 
        }
        gradloc=Axloc;
           
        
        project(gradloc,xloc);
        
        

        res_norm=frob_norm(gradloc);
        lambda=innerprod(Axloc,xloc);
        Index i,j;
        //if(currit%3==4){
        ALSres(i&0)=A(i/2,j/2)*x(j&0)-x(i&0)*lambda;
        /*
        ALSres=TTTensor::random(x.dimensions,maxranks);
        getRes(A,x,lambda,ALSres);*/
        //}
        if (simple){
            XERUS_LOG(info,std::setprecision(4)<<frob_norm(ALSres)<<std::setw(12)<<"\t"<<res_norm<<std::setw(12)<<"\t"<<std::setprecision(12)<<lambda/*- 28.1930439210 +38.979392539208*/<<std::setw(12)<<"\t"<<currit<<std::setw(12)<<"\t"<<(clock()-begin_time)*1e-7/35<<std::setw(12)<<"\t"<<(time(NULL)-begin_time2));
            currit++;
        }else{
        	XERUS_LOG(info,std::setprecision(4)<<frob_norm(ALSres)<<std::setw(12)<<"\t"<<res_norm<<std::setw(12)<<"\t"<<std::setprecision(12)<<lambda- 28.1930439210 +38.979392539208<<std::setw(12)<<"\t"<<"subspace"<<std::setw(12)<<"\t"<<(clock()-begin_time)*1e-7/35<<std::setw(12)<<"\t"<<(time(NULL)-begin_time2));
            
        }
        if (calcresidual){
            TTTensor residual=TTTensor::random(x.dimensions,{250});
            getRes(A,x,lambda,residual);
            std::cout<<"residualnorm: "<<frob_norm(residual)<<std::endl;
        }
        using xerus::misc::operator<<;
        
        XERUS_LOG(JacobiDavidson, "Iteration: " << currit  <<  " EVerr " << lambda/* - 28.1930439210 +38.979392539208 */<< " Residual = " << frob_norm(ALSres) );
        XERUS_LOG(info,"x = " << x.ranks());
            
	}
	
	
	void adapt_ranks(){
        
        if((lambda<switchlambda)&&(currit%8==0)){
            /*if(currit>5){
                auto tmp=TTTensor::random(x.dimensions,10);
                x+=(1e-7)*tmp/frob_norm(tmp);
                maxrankfactor++;
            }
            x.round(10*maxrankfactor);
            
         //   x=local1.retraction_ALS(xloc);
        }else if(currit<5){*/
            x=local1.retraction_ALS(xloc);
        }else{
//             x.round(ranks);
            x=local1.retraction_ALS(xloc);
        }
        using xerus::misc::operator<<;
        std::cout<<x.ranks()<<std::endl;
        
    }
    
};
