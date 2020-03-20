#include <xerus.h>
#include <iostream>
#include <fstream>

using namespace xerus;

class InternalSolver {
	const size_t d;
	
	std::vector<Tensor> leftAStack;
	std::vector<Tensor> rightAStack;

	TTTensor& x;
	const TTOperator& A;


public:
    

	
	InternalSolver(const TTOperator& _A, TTTensor& _x) 
		: d(_x.degree()), x(_x), A(_A)
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
        std::cout<<"4"<<std::endl;
		// Build right stack
        for(size_t currit=0;currit<1000;currit++){
            double lambda=5;
            Index i1, i2, i3, j1 , j2, j3, k1, k2,i,j;
            for (size_t corePosition = 0; corePosition < d; ++corePosition) {
                Tensor op;
                std::cout<<corePosition <<std::endl;
                const Tensor &Ai = A.get_component(corePosition);
                std::cout<<corePosition <<std::endl;
                std::cout<<leftAStack.size() <<std::endl;
                std::cout<<rightAStack.size() <<std::endl;   
                using xerus::misc::operator<<;
                std::cout<<leftAStack.back().dimensions <<std::endl;
                
                std::cout<<rightAStack.back().dimensions<<std::endl;
                std::cout<<Ai.dimensions<<std::endl;
                
                op(i1, i2, i3, j1, j2, j3) = leftAStack.back()(i1, k1, j1)*Ai(k1,i2, j2, k2)*rightAStack.back()(i3, k2, j3);
                std::cout<<corePosition <<std::endl;
            
            
                std::cout<<corePosition <<std::endl;
                lambda = xerus::get_smallest_eigenpair_iterative(x.component(corePosition),op, false, 50000, 1e-8);
                std::cout<<"5"<<std::endl;
                
            
                
                
                
                if (corePosition+1 < d) {
                    x.move_core(corePosition+1, true);
                    push_left_stack(corePosition);
                    rightAStack.pop_back();
                }
                std::cout<<"5"<<std::endl;
            }
                
            std::cout<<"5"<<std::endl;
            // Sweep Right -> Left : only move core and update stacks
            x.move_core(0, true);
            for (size_t corePosition = d-1; corePosition > 0; --corePosition) {

                push_right_stack(corePosition);
                leftAStack.pop_back();
            }
            std::cout<<"6"<<std::endl;
            TTTensor ALSres;
                
                ALSres(i&0)=A(i/2,j/2)*x(j&0)-x(i&0)*lambda;
                 using xerus::misc::operator<<;
        
                XERUS_LOG(simpleALS, "Iteration: " << currit  <<  " EV " << lambda/* - 28.1930439210 +38.979392539208 */<< " Residual = " << frob_norm(ALSres) );
                XERUS_LOG(info,"x = " << x.ranks());    
            
        }
        
    }
	
};


int main(){
    size_t order=10;
    size_t dimension=28;
    std::vector<size_t> dims1=std::vector<size_t>(order, dimension);
    std::vector<size_t> dims2=std::vector<size_t>(order*2, dimension);
    //hier startraenge angeben
    std::cout<<"1"<<std::endl;
    TTTensor  x =TTTensor::random(dims1,50);
    TTOperator A;
    
    std::ifstream read("henon_heiles_10_28_011.ttoperator");
    misc::stream_reader(read,A,xerus::misc::FileFormat::BINARY);
    std::cout<<"2"<<std::endl;
    InternalSolver help(A,x);
    std::cout<<"3"<<std::endl;
    help.solve();
            
            
    

}



