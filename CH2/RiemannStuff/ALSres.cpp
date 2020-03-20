#include <xerus.h>



using namespace xerus;

class InternalResSolver {
	const size_t d;
	

	
	std::vector<Tensor> leftBStack;
	std::vector<Tensor> rightBStack;
	
	TTTensor& x;
	const TTOperator& A;
	const TTTensor& b;


public:
	size_t maxIterations;
	
	InternalResSolver(const TTOperator& _A, TTTensor& _x, const TTTensor& _b) 
		: d(_x.degree()), x(_x), A(_A), b(_b), maxIterations(1)
	{ 

		leftBStack.emplace_back(Tensor::ones({1,1,1}));
		rightBStack.emplace_back(Tensor::ones({1,1,1}));
	}
	
	
	void push_left_stack(const size_t _position) {
		Index i1, i2, i3, j1 , j2, j3, k1, k2;
		const Tensor &xi = x.get_component(_position);
		const Tensor &Ai = A.get_component(_position);
		const Tensor &bi = b.get_component(_position);
		
		Tensor tmpA, tmpB;
		tmpB(i1, i2, i3) = leftBStack.back()(j1, j2, j3)
				*xi(j1, k1, i1)*Ai(j2,k1,k2,i2)*bi(j3, k2, i3);
		leftBStack.emplace_back(std::move(tmpB));
	}
	
	
	void push_right_stack(const size_t _position) {
		Index i1, i2, i3, j1 , j2, j3, k1, k2;
		const Tensor &xi = x.get_component(_position);
		const Tensor &Ai = A.get_component(_position);
		const Tensor &bi = b.get_component(_position);
				
		Tensor tmpA, tmpB;
		tmpB(i1, i2 , i3) = xi(i1, k1, j1)*Ai(i2,k1,k2,j2)*bi(i3, k2, j3)
				*rightBStack.back()(j1, j2, j3);
		rightBStack.emplace_back(std::move(tmpB));
	}
	
	
	
	void solve() {
		// Build right stack
		x.move_core(0, true);
		for (size_t pos = d-1; pos > 0; --pos) {
			push_right_stack(pos);
		}
		
		Index i1, i2, i3, j1 , j2, j3, k1, k2, l1,l2;
		
		
		for (size_t itr = 0; itr < maxIterations; ++itr) {
			
			
			
			// Sweep Left -> Right
			for (size_t corePosition = 0; corePosition < d; ++corePosition) {
				Tensor  rhs;
				
				const Tensor &Ai = A.get_component(corePosition);
				const Tensor &bi = b.get_component(corePosition);
				

				rhs(i1, i2, i3) =   leftBStack.back()(i1, k1, k2) *  Ai(k1,i2,j2,l1)* bi(k2, j2, l2) *   rightBStack.back()(i3, l1,l2);
				x.component(corePosition)=rhs;
				//xerus::solve(x.component(corePosition), op, rhs);
				
				if (corePosition+1 < d) {
					x.move_core(corePosition+1, true);
					push_left_stack(corePosition);

					rightBStack.pop_back();
				}
			}
			
			
			// Sweep Right -> Left : only move core and update stacks
			x.move_core(0, true);
			for (size_t corePosition = d-1; corePosition > 0; --corePosition) {
				push_right_stack(corePosition);

				leftBStack.pop_back();
			}
			
		}

		
	}
	
	
};



void getRes(const TTOperator& A,const TTTensor& x,double lambda, TTTensor& res){
    time_t begin_time = time (NULL);
    using xerus::misc::operator<<;
    TTOperator op=TTOperator::identity(A.dimensions);
    op=A-lambda*op;
    InternalResSolver help(op,res,x);
    
    help.solve();
    /*
    TTTensor test;
    Index i,j;
    test(i&0)=A(i/2,j/2)*x(j&0)-lambda*x(i&0);
    */
    std::cout<<"time to calc res-norm: "<<time (NULL)-begin_time<<std::endl;
}

void testgetRes(){
    size_t order =10;
    size_t dimension =2;
    TTOperator I=TTOperator::identity(std::vector<size_t>(2*order, dimension));
    TTTensor ttx=TTTensor::random(std::vector<size_t>(order, dimension),{3});
    ttx/=frob_norm(ttx);
    TTTensor tty=TTTensor::random(std::vector<size_t>(order, dimension),{3});
    tty/=frob_norm(tty);
    getRes(I,ttx,1,tty);
    std::cout<<"hier sollte eine null stehen: "<<frob_norm(tty)<<std::endl;
}
