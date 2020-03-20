#include <xerus.h>

using xerus::misc::operator<<;


using namespace xerus;

class InternalResSolver {
	const size_t d;



	std::vector<Tensor> leftAStack;
	std::vector<Tensor> leftMStack;
	std::vector<Tensor> rightAStack;
	std::vector<Tensor> rightMStack;

	TTTensor& x;
	const TTOperator& A;
	const TTOperator& M;
	const TTTensor& b;
  const double lambda;

public:
	size_t maxIterations;

	InternalResSolver(const TTOperator& _A,const TTOperator& _M, const double _l, TTTensor& _x, const TTTensor& _b)
		: d(_x.order()), x(_x), A(_A), M(_M), lambda(_l), b(_b), maxIterations(1)
	{

		leftAStack.emplace_back(Tensor::ones({1,1,1,1,1}));
		rightAStack.emplace_back(Tensor::ones({1,1,1,1,1}));
		leftMStack.emplace_back(Tensor::ones({1,1,1,1}));
		rightMStack.emplace_back(Tensor::ones({1,1,1,1}));
	}


	void push_left_stack(const size_t _position) {
		Index i1, i2, i3,i4,i5, j1 , j2, j3,j4,j5, k1, k2,k3,k4;
		const Tensor &xi = x.get_component(_position);
		const Tensor &Ai = A.get_component(_position);
		const Tensor &Mi = M.get_component(_position);
		const Tensor &bi = b.get_component(_position);

		Tensor tmpA, tmpM;
		tmpA(i1, i2, i3,i4,i5) = leftAStack.back()(j1, j2, j3, j4, j5)
				*xi(j1, k1, i1) * Mi(j2,k1,k2,i2)*Ai(j3,k2,k3,i3)*Mi(j4,k3,k4,i4)*bi(j5, k4, i5);
		leftAStack.emplace_back(std::move(tmpA));
		tmpM(i1, i2, i3,i4) = leftMStack.back()(j1, j2, j3, j4)
				*xi(j1, k1, i1) * Mi(j2,k1,k2,i2)*Mi(j3,k2,k3,i3)*bi(j4, k3, i4);
		leftMStack.emplace_back(std::move(tmpM));
	}


	void push_right_stack(const size_t _position) {
		Index i1, i2, i3,i4,i5, j1 , j2, j3,j4,j5, k1, k2,k3,k4;
		const Tensor &xi = x.get_component(_position);
		const Tensor &Ai = A.get_component(_position);
		const Tensor &Mi = M.get_component(_position);
		const Tensor &bi = b.get_component(_position);

		Tensor tmpA, tmpM;
		tmpA(i1, i2, i3,i4,i5) = xi(i1, k1, j1)*Mi(i2,k1,k2,j2)*Ai(i3,k2,k3,j3)*Mi(i4,k3,k4,j4)*bi(i5, k4, j5)
				*rightAStack.back()(j1, j2, j3,j4,j5);
		rightAStack.emplace_back(std::move(tmpA));
		tmpM(i1, i2, i3,i4) = xi(i1, k1, j1)*Mi(i2,k1,k2,j2) * Mi(i3,k2,k3,j3)*bi(i4, k3, j4)
				*rightMStack.back()(j1, j2, j3,j4);
		rightMStack.emplace_back(std::move(tmpM));
	}



	void solve() {
		// Build right stack
		x.move_core(0, true);
		for (size_t pos = d-1; pos > 0; --pos) {
			push_right_stack(pos);
		}

		Index i1, i2, i3,i4,i5, j1 , j2, j3,j4,j5, k1, k2,k3,k4,l1,l2,l3,l4;


		for (size_t itr = 0; itr < maxIterations; ++itr) {



			// Sweep Left -> Right
			for (size_t corePosition = 0; corePosition < d; ++corePosition) {
				Tensor  rhsA, rhsM;

				const Tensor &Ai = A.get_component(corePosition);
				const Tensor &bi = b.get_component(corePosition);
				const Tensor &Mi = M.get_component(corePosition);

				rhsA(i1, i2, i3) =   leftAStack.back()(i1, k1, k2,k3,k4) * Mi(k1,i2,j2,l1)*  Ai(k2,j2,j3,l2)* Mi(k3,j3,j4,l3)* bi(k4, j4, l4) *   rightAStack.back()(i3, l1,l2,l3,l4);
				rhsM(i1, i2, i3) =   leftMStack.back()(i1, k1, k2,k3) * Mi(k1,i2,j2,l1)* Mi(k2,j2,j3,l2)* bi(k3, j3, l3) *   rightMStack.back()(i3, l1,l2,l3);
				x.component(corePosition)=rhsA - lambda*rhsM;
				//xerus::solve(x.component(corePosition), op, rhs);

				if (corePosition+1 < d) {
					x.move_core(corePosition+1, true);
					push_left_stack(corePosition);

					rightAStack.pop_back();
					rightMStack.pop_back();
				}
				//if (corePosition == d - 1){
//					XERUS_LOG(info,"res = " << x.get_component(corePosition).frob_norm());
				//}
			}

//			// Sweep Right -> Left : only move core and update stacks
//			x.move_core(0, true);
//			for (size_t corePosition = d-1; corePosition > 0; --corePosition) {
//				push_right_stack(corePosition);
//
//				leftAStack.pop_back();
//				leftMStack.pop_back();
//			}
		}
	}
};



void getRes(const TTOperator& A,const TTTensor& x, const TTOperator M, double lambda, TTTensor& res){
    time_t begin_time = time (NULL);
    InternalResSolver help(A,M,lambda, res,x);

    help.solve();
    /*
    TTTensor test;
    Index i,j;
    test(i&0)=A(i/2,j/2)*x(j&0)-lambda*x(i&0);
    */
    //XERUS_LOG(info,"time to calc res-norm: "<<time (NULL)-begin_time);
}
//
//void testgetRes(){
//    size_t order =10;
//    size_t dimension =2;
//    TTOperator I=TTOperator::identity(std::vector<size_t>(2*order, dimension));
//    TTTensor ttx=TTTensor::random(std::vector<size_t>(order, dimension),{3});
//    ttx/=frob_norm(ttx);
//    TTTensor tty=TTTensor::random(std::vector<size_t>(order, dimension),{3});
//    tty/=frob_norm(tty);
//    getRes(I,ttx,1,tty);
//    std::cout<<"hier sollte eine null stehen: "<<frob_norm(tty)<<std::endl;
//}
