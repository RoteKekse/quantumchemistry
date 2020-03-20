#include <xerus.h>
#include "normalJD.cpp"


using namespace xerus;

class InternalSolver {
	const size_t d;
	
	std::vector<Tensor> leftAStack;
	std::vector<Tensor> rightAStack;

	TTTensor& x;
	const TTOperator& A;
    double target_lambda;

public:
    
    size_t subspacesteps;
	size_t maxIterations;
	
	InternalSolver(const TTOperator& _A, TTTensor& _x, double targetlambda) 
		: d(_x.degree()), x(_x), A(_A),target_lambda(targetlambda),maxIterations(1)
	{
        x/=frob_norm(x);
		leftAStack.emplace_back(Tensor::ones({1,1,1}));
		rightAStack.emplace_back(Tensor::ones({1,1,1}));
        
        x.move_core(0, true);
		for (size_t pos = d-1; pos > 0; --pos) {
			push_right_stack(pos);
		}

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
	

	
	void solve() {
		// Build right stack

		time_t begin_time=time(NULL);
		Index i1, i2, i3, j1 , j2, j3, k1, k2,i,j;
        for (size_t corePosition = 0; corePosition < d; ++corePosition) {
            Tensor op;
            
            const Tensor &Ai = A.get_component(corePosition);
	
				
            op(i1, i2, i3, j1, j2, j3) = leftAStack.back()(i1, k1, j1)*Ai(k1,i2, j2, k2)*rightAStack.back()(i3, k2, j3);
            
        
           
            
            targeteigen(op,x.component(corePosition),maxIterations,0.000000000000001,target_lambda);
            
            std::cout<<"time for ALS, corePosition: "<< corePosition<<" "<<time (NULL)-begin_time<<" sekunden"<<std::endl;
            
            
           
            
            
            
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
	
};


void doALS(TTTensor& x, const TTOperator& A, std::ofstream& myfile,double target_lambda,size_t maxIterations){
		Index i,j,k;
        time_t begin_time=time(NULL); 
        for(size_t it=0;it<maxIterations;it++){
            value_t error;
            double lambda;

            localProblem local(x);
            auto locx=local.getlocalVector(x);
            Tensor lam;
            local.projection(locx);
            auto Ax=local.localProduct(locx,A);
            lam()=Ax(i)*locx(i);
        
             lambda=lam[0];
        
            TTTensor res=TTTensor::random(x.dimensions,x.ranks());
            
            getRes(A,x,lambda,res);
        
            
            error=frob_norm(res/lambda);

        
        
            myfile<<error<<std::setw(12)<<"\t"<<lambda<<std::setw(12)<<"\t"<<it<<std::setw(12)<<"\t"<<(time(NULL)-begin_time)<<std::endl;
            InternalSolver help(A,x,target_lambda);
            help.solve();
            
            
        }

	}



