#include <xerus.h>


using namespace xerus;


class vector_transport{
	const size_t d;
	std::vector<size_t> ranks;
	TTTensor& x;

	const TTTensor& b;
	std::vector<Tensor> leftBStack;
	std::vector<Tensor> rightBStack;

public:
	TTTensor trans;
		
	vector_transport(TTTensor& _x, const TTTensor& _b)
	:d(_x.degree()),x(_x),b(_b){
		ranks=x.ranks();
		for (size_t pos=0;pos<d-1;pos++){
			ranks[pos]*=2;
		}
		trans=TTTensor(x.dimensions);
		leftBStack.emplace_back(Tensor::ones({1,1}));
		rightBStack.emplace_back(Tensor::ones({1,1}));
		transport();

	}

	
	
	void push_left_stack(const size_t _position) {
		Index i1, i2, i3, j1 , j2, j3, k1, k2;
		const Tensor &xi = x.get_component(_position);

		const Tensor &bi = b.get_component(_position);
		
		Tensor  tmpB;
	
		tmpB(i1, i2) = leftBStack.back()(j1, j2)
				*xi(j1, k1, i1)*bi(j2, k1, i2);
		leftBStack.emplace_back(std::move(tmpB));
	}



	void push_right_stack(const size_t _position) {
		Index i1, i2, i3, j1 , j2, j3, k1, k2;
		const Tensor &xi = x.get_component(_position);
		
		const Tensor &bi = b.get_component(_position);
		
		Tensor  tmpB;
		
		
		tmpB(i1, i2) = xi(i1, k1, j1)*bi(i2, k1, j2)
				*rightBStack.back()(j1, j2);
		rightBStack.emplace_back(std::move(tmpB));
	}
	
	void transport(){
		// Build right stack
		x.move_core(0, true);
		for (size_t pos = d-1; pos > 0; --pos) {
			push_right_stack(pos);
		}
		
		Index i1, i2, i3,j1,j2,j3, k1, k2;
		// Sweep Left -> Right
		for (size_t corePosition = 0; corePosition < d; ++corePosition) {
			Tensor rhs;
				

			const Tensor &bi = b.get_component(corePosition);
				

			rhs(i1, i2, i3) = leftBStack.back()(i1, k1) * bi(k1, i2, k2) * rightBStack.back()(i3, k2);
			TTTensor comp=x;
				

				
			if (corePosition+1 < d) {
				
				
				x.move_core(corePosition+1, true);
				Tensor q= x.get_component(corePosition);
				rhs(i1,i2,i3)=rhs(i1,i2,i3)-q(i1,i2,k1)*q(j1,j2,k1)*rhs(j1,j2,i3);
				push_left_stack(corePosition);
				
				rightBStack.pop_back();
			}
			comp.set_component(corePosition,rhs);
		
			trans+=comp;
			trans.round(ranks);
		}
		
		

	}
	

};


