#include <xerus.h>

using namespace xerus;

class normalJD {
	const size_t d;
	

	Tensor& x;
	const Tensor& A;
    double target_lambda;
    
    
    
    
    public:

        
    size_t CG_minit=10;
    size_t CG_maxit=30;
    value_t CG_error=0.0001;
    
    size_t eigenvector_steps=100;    
	
	normalJD(const Tensor& _A, Tensor& _x, double targetlambda) 
		: d(_x.degree()), x(_x), A(_A),target_lambda(targetlambda)
	{
        x/=frob_norm(x);
		
	}
	
	
	void simple(){
        auto dir=getdirection();
        x-=dir;
        x/=frob_norm(x);
        
    }
	void subspace(size_t maxit){
        Index i,j,k;
        
       
        std::vector<Tensor> V;
        V.emplace_back(x);
        std::vector<TTTensor> AV;
        Tensor h;
        h(i&0)=A(i/2,j/2)*x(j&0);
        AV.emplace_back(h);
        auto H=Tensor({V.size(),V.size()});
        h()=h(i&0)*x(i&0);
        H[{0,0}]=h[0];
        Tensor eigenvector=Tensor::dirac({1},{0});
        for (size_t it=0;it<maxit;it++){
            //writelessData();
            
            //std::cout<<Aloc.to_string()<<std::endl;
            
        //newdirection
            Tensor v=getdirection();
		//orthonormalize against V
			for (size_t pos=0;pos<V.size();pos++){
				Tensor h;
				h()=v(j&0)*V[pos](j&0);
				v-=h[0]*V[pos];
				
			}
			
			v/=frob_norm(v);
			V.emplace_back(v);
		//built A*V
			
			
            Tensor h;
            h(i&0)=A(i/2,j/2)*v(j&0);
            AV.emplace_back(h);
			
			
		//built V^T*A*V
			H.resize_mode(0,V.size(),H.dimensions[0]);
			H.resize_mode(1,V.size(),H.dimensions[1]);
			for (size_t l=0;l<V.size();l++){
				
					Tensor h;
					
                    h()=AV[l](j&0)*v(j&0);
                   
					H[{l,V.size()-1}]=h[0];
                    H[{V.size()-1,l}]=h[0];
                    
				
			}
            //std::cout<<H.to_string()<<std::endl;

			eigenvector.resize_mode(0,it+2,it+1);
		//target eigenvector, else maxeigenvector
			
				targeteigen
				(H,eigenvector,eigenvector_steps,0.00000000000001,target_lambda-0.001);
			
		//built x
           // std::cout<<eigenvector.to_string()<<std::endl;
			Tensor xnew=Tensor(x.dimensions);
			for (size_t pos=0;pos<V.size();pos++){
				xnew+=eigenvector[pos]*V[pos];
                
			}
			
			x=xnew/frob_norm(xnew);
            outputdata();
        }

        
        
            
        
        
    }
    
    Tensor getdirection(){
        Index i,j,k,l,m;
            Tensor P=Tensor::identity(A.dimensions);
            P(i/2,j/2)=P(i/2,j/2)-x(i&0)*x(j&0);
            /*
             * maybe for other eigenvectors
            for (size_t n=0;n<eigenvectors.size();n++){
                Tensor q=local1.getlocalVector(eigenvectors[n]);
                q(i)=P(i,j)*q(j);
                P(i,j)=P(i,j)-q(i)*q(j);
            }
            */
            Tensor Op, I, direction,rhs;
            I=Tensor::identity(P.dimensions);
            Op(i/2,j/2)=P(i/2,k/2)*(A(k/2,l/2)-target_lambda*I(k/2,l/2))*P(l/2,j/2);
            rhs(i&0)=P(i/2,j/2)*A(j/2,k/2)*x(k&0);
            direction=rhs;
            cg_method(Op,P,rhs,direction,CG_maxit,CG_minit,CG_error);
            //xerus::solve(direction,Op,rhs);
            direction(i&0)=P(i/2,j/2)*direction(j&0);
            return direction;
        
    }
    
    void outputdata(){
        Index i,j,k;
        Tensor Ax,lam;
        Ax(i&0)=A(i/2,j/2)*x(j&0);
        lam()=Ax(i&0)*x(i&0);
        std::cout<<"error: "<<frob_norm(Ax/lam[0]-x)<<"\t"<<"lambda: "<<lam[0]<<std::endl;
    }
};

void trynormalJD(){
    
    Index i,j,k;
    auto A=Tensor::random({6,2,6,6,2,6});
    A(i/2,j/2)=A(i/2,k/2)*A(j/2,k/2);
    auto x=Tensor::random({6,2,6});
    
    normalJD help(A,x,0);
    for (size_t it=0;it<100;it++){
        Tensor Ax,lam;
        Ax(i&0)=A(i/2,j/2)*x(j&0);
        lam()=Ax(i&0)*x(i&0);
        std::cout<<"error: "<<frob_norm(Ax/lam[0]-x)<<"\t"<<"lambda: "<<lam[0]<<std::endl;
        help.subspace(10);
        
        
    }
    
    
}

	
	
