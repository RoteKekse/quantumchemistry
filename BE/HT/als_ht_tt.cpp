

#include <xerus.h>

using namespace xerus;
using xerus::misc::operator<<;

class InternalSolver {
	const size_t d;
	const size_t p;

	std::vector<Tensor> leftAStack;
	std::vector<TensorNetwork> rightAStack;

	std::vector<Tensor> leftBStack;
	std::vector<Tensor> rightBStack;

	HTTensor& x;
	const TTOperator& A;
	const TTTensor& b;
	const double solutionsNorm;
public:
	size_t maxIterations;

	InternalSolver(const TTOperator& _A, HTTensor& _x, const TTTensor& _b)
		: d(_x.degree()), x(_x), A(_A), b(_b), solutionsNorm(frob_norm(_b)), maxIterations(1000), p(static_cast<size_t>(std::log(_x.degree())))
	{
	}


	void push_left_stack(const size_t _position) {
		Index i1, i2, i3, j1 , j2, j3, k1, k2;
		const Tensor &xi = x.get_component(_position);
		const Tensor &Ai = A.get_component(_position);
		const Tensor &bi = b.get_component(_position);

		Tensor tmpA, tmpB;
		tmpA(i1, i2, i3) = leftAStack.back()(j1, j2, j3)
				*xi(j1, k1, i1)*Ai(j2, k1, k2, i2)*xi(j3, k2, i3);
		leftAStack.emplace_back(std::move(tmpA));
		tmpB(i1, i2) = leftBStack.back()(j1, j2)
				*xi(j1, k1, i1)*bi(j2, k1, i2);
		leftBStack.emplace_back(std::move(tmpB));
	}


	void push_right_stack(const size_t _position) {
		Index i1, i2, i3, i4, i5, j1 , j2, j3, j4, j5, k1, k2;
		std::cout << _position << std::endl;
		TensorNetwork tmpA, tmpB,tmpAx,tmpBx;
		if (_position >= d - 1){
			const Tensor &xi = x.get_component(_position);
			const Tensor &Ai = A.get_component(_position-d+1);
			if (_position == 2*d-2){
				auto dummy = Tensor::ones({1});
				tmpA(i1, i2, i3) = xi(i1, k1)*Ai(i2, k1, k2, j2)*xi(i3, k2)*dummy(j2);
			}
			else if (_position == d - 1){
				auto dummy = Tensor::ones({1});
				tmpA(i1, i2, i3) = xi(i1, k1)*Ai(j2, k1, k2, i2)*xi(i3, k2)*dummy(j2);

			}
			else{
				tmpA(i1, i2, i3,i4) = xi(i1, k1)*Ai(i2, k1, k2, i3)*xi(i4, k2);
			}


		}
		else if (_position){

		}
		rightAStack.emplace_back(std::move(tmpAx));
//		TensorNetwork tmpA, tmpB;
//		tmpA(i1, i2, i3) = xi(i1, k1, j1)*Ai(i2, k1, k2, j2)*xi(i3, k2, j3)
//				*rightAStack.back()(j1, j2, j3);
//		rightAStack.emplace_back(std::move(tmpA));
//		tmpA.draw(std::to_string(_position) + ".svg");
//		tmpB(i1, i2) = xi(i1, k1, j1)*bi(i2, k1, j2)
//				*rightBStack.back()(j1, j2);
//		rightBStack.emplace_back(std::move(tmpB));
	}

	double calc_residual_norm() {
		Index i,j;
		return frob_norm(A(i/2, j/2)*x(j&0) - b(i&0)) / solutionsNorm;
	}

	std::vector<size_t> calc_traversal(){
		std::vector<size_t> trav;
		trav.emplace_back(2*d-2);
		auto elm = trav.back();
		while (trav.size() < 2*d-1){
				if (std::find(trav.begin(), trav.end(), 2*elm + 2) == trav.end() && 2*elm + 2 < 2*d - 1){
									elm = 2*elm + 2;
				}
				else if (std::find(trav.begin(), trav.end(), 2*elm + 1) == trav.end() && 2*elm + 1 < 2*d - 1){
									elm = 2*elm + 1;
				}
				else if (std::find(trav.begin(), trav.end(), elm) != trav.end()){
					elm = (elm-1) / 2;
					if(std::find(trav.begin(), trav.end(), elm) == trav.end()){
						trav.emplace_back(elm);
					}
				}
				else{
					trav.emplace_back(elm);
				}
		}
		return trav;
	}

	void solve() {
		// Build right stack
		auto trav = calc_traversal();
//		for (size_t pos: trav) {
//			push_right_stack(pos);
//		}
//
		Index i1, i2, i3, j1 , j2, j3, k1, k2;
		std::vector<double> residuals(10, 1000.0);
//
		for (size_t itr = 0; itr < maxIterations; ++itr) {
			// Calculate residual and check end condition
			residuals.push_back(calc_residual_norm());
			if (itr > 10 && residuals.back()/residuals[residuals.size()-10] > 0.99999) {
				XERUS_LOG(simpleALS, "Done! Residual decrease from " << std::scientific << residuals[10] << " to " << std::scientific << residuals.back() << " in " << residuals.size()-10 << " iterations.");
				return; // We are done!
			}
			XERUS_LOG(simpleALS, "Iteration: " << itr << " Residual: " << residuals.back());

			// Sweep Left -> Right
			for (size_t corePosition : trav) {
				if (corePosition == 0 || corePosition >= d-1)
					continue;
				x.move_core(corePosition, true);

				TensorNetwork optn, rhstn;
				Tensor op, rhs;
				optn() =   x(i1^d)*x(j1^d) *  A(i1^d,j1^d);
				optn.nodes[corePosition].erase();
				optn.nodes[corePosition+2*d].erase();
				rhstn() =   x(j1^d) *  b(j1^d);
				rhstn.nodes[corePosition].erase();
				//optn.draw("op.svg");
				//rhstn.draw("rhs.svg");
				if (corePosition == 0) {
					rhstn.nodes[2*d-1].erase();
					optn.nodes[2*d-1].erase();
					optn.nodes[2*d-1 + 2*d].erase();
				}
				rhstn.sanitize();
				optn.sanitize();

				if (corePosition >= d - 1 || corePosition == 0){
					op (i1,j1,i2,j2) = optn(i1,i2,j1,j2);
					rhs (i1,i2) = rhstn(i1,i2);

				} else {

					op (i1,i2,i3,j1,j2,j3) = optn(i1,i2,i3,j1,j2,j3);
					rhs (i1,i2,i3) = rhstn(i1,i2,i3);
				}
//				XERUS_LOG(info,"op    = " << op.dimensions);
//				XERUS_LOG(info,"rhs   = " << rhs.dimensions);
//				XERUS_LOG(info,"optn  = " << optn.dimensions);
//				XERUS_LOG(info,"rhstn = " << rhstn.dimensions);

				Tensor &xi = x.component(corePosition);
				if (corePosition == 0) {
					xi.reinterpret_dimensions({xi.dimensions[1],xi.dimensions[2]});
				}
				xerus::solve(xi, op, rhs);
				if (corePosition == 0) {
					xi.reinterpret_dimensions({1,xi.dimensions[0],xi.dimensions[1]});
				}
				x.set_component(corePosition,xi);
//

			}
		}
	}

};

void simpleALS(const TTOperator& _A, HTTensor& _x, const TTTensor& _b)  {
	InternalSolver solver(_A, _x, _b);
	return solver.solve();
}


int main() {
	Index i,j,k;
	auto dim = 16;
	auto s = 3;
	auto A = TTOperator::random(std::vector<size_t>(2*dim, s), std::vector<size_t>(dim-1,2));
	A(i/2,j/2) = A(i/2, k/2) * A(j/2, k/2);

	auto solution = HTTensor::random(std::vector<size_t>(dim, s), std::vector<size_t>(2*dim-2,3));
	TTTensor b;
	b(i&0) = A(i/2, j/2) * solution(j&0);



	auto x = HTTensor::random(std::vector<size_t>(dim, s), std::vector<size_t>(2*dim-2,3));
	simpleALS(A, x, b);
//
	XERUS_LOG(info, "Residual: " << frob_norm(A(i/2, j/2) * x(j&0) - b(i&0))/frob_norm(b));
	XERUS_LOG(info, "Error: " << frob_norm(solution-x)/frob_norm(x));
}

