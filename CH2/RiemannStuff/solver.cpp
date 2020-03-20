#include <xerus.h>
#include <time.h>
#include <cmath>
#include <iostream>
#include <fstream>
#include "localProblem.cpp"
#include "eigen.cpp"
#include "gradient.cpp"
#include "vectorTransport.cpp"
#include "cgmethod.cpp"
#include "ALSres.cpp"



using namespace xerus;
using xerus::misc::operator<<;


class solver{
	TTTensor& x;
	const TTOperator& A;
    Tensor Aloc;
    Tensor xloc;
	size_t currit;
	std::vector<size_t> ranks;
    
	std::ofstream& myfile;
    localProblem local1;
    bool rankadaption;
    
    
    
public:
    std::vector<size_t> maxranks;
    value_t epsilon;
    
    bool targetadaption;
    std::vector<TTTensor> eigenvectors;
    int method=1;
    bool target;
    bool loc;
    double target_lambda;
    std::vector<double> lambdaold;
    double lambda;
    double switchlambda;
    value_t error;	
    value_t error2;	
    size_t eigenvector_steps=10;
    size_t CG_minit=15;
    size_t CG_maxit=30;
    value_t CG_error=0.0000001;
    value_t targetadaptionconst=0.0000001;
    value_t ranktoerror=0.00000000000001;
    value_t rankratio=3;
    time_t begin_time;
    
    
   solver(const TTOperator& _A, TTTensor& _x, std::ofstream& file) 
	:local1(x),  x(_x), A(_A),myfile(file)
	{ 
        currit=0;
        using xerus::misc::operator<<;
        switchlambda=-100000000000000;
        myfile<<A.ranks()<<std::endl;
		myfile<<"target eigenvalue: max"<<std::endl<<std::endl;
	//	correct_eq_solver ;		
		target=false;		
        rankadaption=false;
        targetadaption=false;
		ranks=x.ranks();
		x/=frob_norm(x);


        begin_time = time (NULL);
        
	}

	solver(const TTOperator& _A, TTTensor& _x,double _targetlambda,std::ofstream& file) 
	:local1(x),  x(_x), A(_A),myfile(file)
	{ 
        std::cout<<" "<<std::endl;
        currit=0;
        using xerus::misc::operator<<;
        
        switchlambda=-100000000000000;
        myfile<<A.ranks()<<std::endl;
        myfile<<x.ranks()<<std::endl;
		std::cout<<" "<<std::endl;
		target =true;
        rankadaption=false;
        targetadaption=false;
		target_lambda=_targetlambda;		
		myfile<<"target eigenvalue: "<<target_lambda<<std::endl<<std::endl;	
		ranks=x.ranks();
		x/=frob_norm(x);


        begin_time =time (NULL);
	}
    
    void tangentspacesimple(size_t maxit){
        
        Index i,j,k;
        local1.update(x);
        xloc=local1.getlocalVector(x);
        xloc(i)=local1.projectionT(i,j)*xloc(j);
        Aloc=local1.getlocalOperator(A);
        writeData();
        for (size_t it=0;it<maxit;it++){
            //writelessData();
            Tensor dir=getdirection();
            xloc-=dir;
            xloc/=frob_norm(xloc);
            
        }
        x=local1.builtTTTensor(xloc);
        if (rankadaption)
            x+=0.0000001*TTTensor::random(x.dimensions,{1});
        adapt_ranks();
        
    }
    
    
    void tangentspacesolve(size_t maxit){
        Index i,j,k;
        local1.update(x);
        xloc=local1.getlocalVector(x);
        local1.projection(xloc);
        Aloc=Tensor();
        Aloc=local1.getlocalOperator(A);
        std::vector<Tensor> V;
        V.emplace_back(xloc);
        std::vector<TTTensor> AV;
        Tensor h;
        h(i)=Aloc(i,j)*xloc(j);
        AV.emplace_back(h);
        auto H=Tensor({V.size(),V.size()});
        h()=h(i)*xloc(i);
        H[{0,0}]=h[0];
        Tensor eigenvector=Tensor::dirac({1},{0});
        writeData();
        for (size_t it=0;it<maxit;it++){
            //writelessData();
            
            //std::cout<<Aloc.to_string()<<std::endl;
            
        //newdirection
            Tensor v=getdirection();
		//orthonormalize against V
			for (size_t pos=0;pos<V.size();pos++){
                
                Tensor h;
				h()=v(j)*V[pos](j);
                
				v-=h[0]*V[pos];
                
            
				
			}
			
            
			v/=frob_norm(v);
			V.emplace_back(v);
		//built A*V
			
			
            Tensor h;
            h(i)=Aloc(i,j)*v(j);
            AV.emplace_back(h);	
			
			
		//built V^T*A*V
			H.resize_mode(0,V.size(),H.dimensions[0]);
			H.resize_mode(1,V.size(),H.dimensions[1]);
			for (size_t l=0;l<V.size();l++){
				
					Tensor h;
					
                    h()=AV[l](j)*v(j);
                   
					H[{l,V.size()-1}]=h[0];
                    H[{V.size()-1,l}]=h[0];
                    
				
			}
            //std::cout<<H.to_string()<<std::endl;

			eigenvector.resize_mode(0,it+2,it+1);
		//target eigenvector, else maxeigenvector
			if (target){
				targeteigen
				(H,eigenvector,eigenvector_steps,0.00000000000001,target_lambda-0.001);
			}else{
				maxeigen
				(H,eigenvector,eigenvector_steps,0.00000000000001);
			}
		//built x
           // std::cout<<eigenvector.to_string()<<std::endl;
			Tensor xnew=Tensor(xloc.dimensions);
			for (size_t pos=0;pos<V.size();pos++){
				xnew+=eigenvector[pos]*V[pos];
                
			}
			
			xloc=xnew;
            
        }
        x=local1.builtTTTensor(xloc);
        if (rankadaption)
            x+=0.0000001*TTTensor::random(x.dimensions,{1});
        adapt_ranks();
            
        
        
    }
    
    
    void simple(){
        Index i,j,k;
        x/=frob_norm(x);
        local1.update(x);
        xloc=local1.getlocalVector(x);
        
       
        //Tensor P=local1.projection;
         /*
         * maybe do something like this for other eigenvectors
         * 
            for (size_t n=0;n<eigenvectors.size();n++){
                Tensor q=local1.getlocalVector(eigenvectors[n]);
                q(i)=P(i,j)*q(j);
                P(i,j)=P(i,j)-q(i)*q(j);
            }
            
        */    
        local1.projection(xloc);
        //std::cout<<xloc.to_string()<<std::endl;
        if(loc){
            Aloc=local1.getlocalOperator(A);
        }
        writeData();
        Tensor v=getdirection();
        xloc-=v;
        x=local1.builtTTTensor(xloc);
        adapt_ranks();
    }


    void subspace( size_t maxit){
        Index i,j,k;
        local1.update(x);
        xloc=local1.getlocalVector(x);
        xloc(i)=local1.projectionT(i,j)*xloc(j);
        std::vector<Tensor> V;
        V.emplace_back(xloc);
        Tensor eigenvector=Tensor::dirac({1},{0});
       
        for (size_t it=0;it<maxit;it++){
            xloc=local1.getlocalVector(x);
            local1.projection(xloc);
            Aloc=local1.getlocalOperator(A);
            //std::cout<<Aloc.to_string()<<std::endl;
            writeData();
        //newdirection
            Tensor v=getdirection();
		//orthonormalize against V
			for (size_t pos=0;pos<V.size();pos++){
				Tensor h;
				h()=v(j)*V[pos](j);
				v-=h[0]*V[pos];
				
			}
			
			v/=frob_norm(v);
			V.emplace_back(v);
		//built A*V
			std::vector<TTTensor> AV;
			for (size_t pos=0;pos<V.size();pos++){
				Tensor h;
				h(i)=Aloc(i,j)*V[pos](j);
				AV.emplace_back(h);	
			}
			
		//built V^T*A*V
			auto H=Tensor({V.size(),V.size()});
			for (size_t l=0;l<V.size();l++){
				for (size_t m=0;m<V.size();m++){
					Tensor h;
					
                    h()=AV[l](j)*V[m](j);
                   
					H[{l,m}]=h[0];
                    
				}
			}
            

			eigenvector.resize_mode(0,it+2,it+1);
		//target eigenvector, else maxeigenvector
			if (target){
				targeteigen
				(H,eigenvector,eigenvector_steps,0.00000000000001,target_lambda-0.001);
			}else{
				maxeigen
				(H,eigenvector,eigenvector_steps,0.00000000000001);
			}
		//built x
           // std::cout<<eigenvector.to_string()<<std::endl;
			Tensor xnew=Tensor(xloc.dimensions);
			for (size_t pos=0;pos<V.size();pos++){
				xnew+=eigenvector[pos]*V[pos];
                
			}
			x=local1.builtTTTensor(xnew);
			adapt_ranks();
            
            localProblem local2(x);

            
			if(it+1<maxit){
		//vector transport V
				for (size_t pos=0;pos<V.size();pos++){
					TTTensor tmp1=local1.builtTTTensor(V[pos]);
                    Tensor tmp2=local2.getlocalVector(tmp1);
                    local1.projection(tmp2);
					V[pos]=tmp2;
                    
				}
		//orthonormalize V
				for (size_t pos=0;pos<V.size();pos++){
					for (size_t pos2=0;pos2<pos;pos2++){
						Tensor h=Tensor::ones({1});
						h(i)=h(i)*V[pos](j&0)*V[pos2](j&0);
						V[pos]-=h[0]*V[pos2];
					}
					V[pos]/=frob_norm(V[pos]);
				}
			}
			local1.update(x);
            
			
            
            
		}
	}
	
	void writelessData(){
    Index i,j,k;
		Tensor grad,lam,P;
		grad(i)=Aloc(i,j)*xloc(j);
        lam()=grad(i)*xloc(i);
        
        //P=local1.projection;
        //P(i,j)=P(i,j)-xloc(i)*xloc(j);
        lambda=lam[0];

        local1.projection(grad);
        Tensor tmp;
        tmp(i)=xloc(i)*xloc(j)*grad(j);
        grad-=tmp;
        error2=frob_norm(grad/lambda);


        
        
        myfile<<"lessData"<<std::setw(12)<<"\t"<<error2<<std::setw(12)<<"\t"<<lambda - 28.1930439210 +38.979392539208<<std::setw(12)<<"\t"<<currit<<std::setw(12)<<"\t"<<(time (NULL)-begin_time)<<std::setw(10)<<"\t"<<targetadaption<<std::setw(6)<<"\t"<<rankadaption<<std::endl;
        myfile<<x.ranks()<<std::endl;
        currit++;

	}
	
	
	void writeData(){
		Index i,j,k;
		Tensor grad,lam,P;
        if(loc){
            grad(i)=Aloc(i,j)*xloc(j);
        }else{
            grad=local1.localProduct(xloc,A);
        }
        lam()=grad(i)*xloc(i);
        
        //P=local1.projection;
        //P(i,j)=P(i,j)-xloc(i)*xloc(j);
        lambda=lam[0];
        lambdaold.emplace_back(lambda);
        local1.projection(grad);
        Tensor tmp;
        tmp(i)=xloc(i)*xloc(j)*grad(j);
        grad-=tmp;
        TTTensor res=TTTensor::random(x.dimensions,ranks);
        getRes(A,x,lambda,res);
        //auto tmpres=getRes(A,x);
        //tmpres/=lambda*lambda;
        //tmpres-=1;
        //error=sqrt(tmpres);

        
        if(lambda<switchlambda){
                targetadaption=true;
        }
        error=frob_norm(res);
        error2=frob_norm(grad);
       /*
        if (error<rankepsilon){
            eigenvectors.emplace_back(x);
            x=TTTensor::random({x.dimensions},{1});
            for(size_t n=0;n<eigenvectors.size();n++){
                Tensor h;
                h()=x(i&0)*eigenvectors[n](i&0);
                x-=h[0]*eigenvectors[n];
                x.round({1});
                ranks=x.ranks();
                x/=frob_norm(x);
                targetadaption=false;
                local1.update(x);
                xloc=local1.getlocalVector(x);
                xloc(i)=local1.projection(i,j)*xloc(j);
                Aloc=local1.getlocalOperator(A);
            }
        }
        */
        if (targetadaption){
            if (lambda<lambdaold[10])
            target_lambda=lambda;
           
                rankadaption=(error/error2>rankratio);
        
                
        }
        
        
        myfile<<std::setprecision(4)<<error<<std::setw(12)<<"\t"<<error2<<std::setw(12)<<"\t"<<std::setprecision(12)<<lambda << "\t"<<std::setprecision(12)<<lambda - 28.1930439210 +38.979392539208<<std::setw(12)<<"\t"<<currit<<std::setw(12)<<"\t"<<(time (NULL)-begin_time)<<std::setw(10)<<"\t"<<targetadaption<<std::setw(6)<<"\t"<<rankadaption<<std::endl;
        myfile<<x.ranks()<<std::endl;
        currit++;

	}
	
	
	Tensor getdirection(){
        if(method==0){
            //Tensor P=local1.projectionS;
            Tensor AX;
            Index i,j,k;
            //AX(i)=P(i,j)*Aloc(j,k)*xloc(k);
            
            return AX;
        }else if(method==1){
            Index i,j,k,l,m;
            Tensor P;
            if(loc){
                local1.built_projection();
                P=local1.projectionT;
                P(i,j)=P(i,j)-xloc(i)*xloc(j);
            }    
            
            
            /*
             * maybe for other eigenvectors
            for (size_t n=0;n<eigenvectors.size();n++){
                Tensor q=local1.getlocalVector(eigenvectors[n]);
                q(i)=P(i,j)*q(j);
                P(i,j)=P(i,j)-q(i)*q(j);
            }
            */
            Tensor Op, I, direction,rhs;
            if (loc){
            I=Tensor::identity(P.dimensions);
            
                Op(i,j)=P(i,k)*(Aloc(k,l)-target_lambda*I(k,l))*P(l,j);
                rhs(i)=P(i,j)*Aloc(j,k)*xloc(k);
                direction=rhs;
                cg_method(Op,P,rhs,direction,CG_maxit,CG_minit,CG_error);
            }else{
                TTOperator Op,I;
                I=TTOperator::identity(A.dimensions);
                Op=A-target_lambda*I;
                rhs=local1.localProduct(xloc,A);
                Tensor tmp;
                tmp(i)=xloc(i)*xloc(j)*rhs(j);
                rhs-=tmp;

                direction=rhs;
                local1.cg_method(Op,xloc,rhs,direction,CG_maxit,CG_minit,CG_error);
                
                
            }
            
            //direction(i)=P(i,j)*direction(j);
            return direction;
        }else if(method==2){
            Index i,j,k,l,m;
            Tensor P=local1.projectionT;
            Tensor Op, I, direction,rhs;
            I=Tensor::identity(P.dimensions);
            Op(i,j)=P(i,k)*(Aloc(k,l)-target_lambda*I(k,l))*P(l,j);
            rhs(i)=P(i,j)*xloc(j);
            direction=rhs;
            cg_method(Op,P,rhs,direction,CG_maxit,CG_minit,CG_error);
            direction(i)=P(i,j)*direction(j);
            return direction;
            
        }else if(method==3){
            
            //preconditioned with blocks
            Index i,j,k,l,m;
            Tensor P=local1.projectionT;
            P(i,j)=P(i,j)-xloc(i)*xloc(j);

            Tensor Op, I, direction,rhs;
            I=Tensor::identity(P.dimensions);
            auto M=local1.getBlockDiagOp(A);
            M(k,l)=(M(k,l)-target_lambda*I(k,l));
            Op(i,j)=P(i,k)*(Aloc(k,l)-target_lambda*I(k,l))*P(l,j);
            rhs(i)=P(i,j)*Aloc(j,k)*xloc(k);
            direction=rhs;
            pcg_method(Op,M,P,rhs,direction,CG_maxit,CG_minit,CG_error
            );
            direction(i)=P(i,j)*direction(j);
            return direction;
            
            
        }
        
    }
    
    void adapt_ranks(){
        using xerus::misc::operator<<; 
       // std::cout<<x.ranks()<<std::endl;
        if (rankadaption){
            auto newranks =ranks;
            for (size_t pos=0;pos<ranks.size();pos++){
                newranks[pos]++;
            }
            x.round(maxranks,epsilon);
            
            
            std::cout<<currit<<" iterationen "<<(time (NULL)-begin_time)<<" sekunden "<<x.ranks()<<std::endl;
            ranks=x.ranks();
            
        }else{
            x.round(ranks,ranktoerror);
            if (ranks!=x.ranks()){
                
        
                std::cout<<currit<<" iterationen "<<(time (NULL)-begin_time)<<" sekunden "<<x.ranks()<<std::endl;
            }
            ranks=x.ranks();
            
        }
        
    }

};
