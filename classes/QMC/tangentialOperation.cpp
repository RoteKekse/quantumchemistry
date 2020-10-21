#include <xerus.h>

using namespace xerus;
using xerus::misc::operator<<;

class TangentialOperation{
public:
    const size_t d;
    std::vector<TTTensor> xbasis;

private:
    Tensor leftAStack;
    std::vector<Tensor> rightAStack;

    Tensor leftxStack;
    std::vector<Tensor> rightxStack;

public:
    TangentialOperation(TTTensor& x) : d(x.order()){
		x.move_core(0,true);
		xbasis.emplace_back(x);
		x.move_core(d-1,true);
		xbasis.emplace_back(x);
  }

	void update(TTTensor& x){
		xbasis.clear();
		x.move_core(0,true);
		xbasis.emplace_back(x);
		x.move_core(d-1,true);
		xbasis.emplace_back(x);
	}

	 TTTensor builtTTTensor(const std::vector<Tensor>& y){
	  	time_t begin_time = time (NULL);
	    TTTensor Y(xbasis[0].dimensions);

			for (size_t pos=0;pos<d;pos++){
				Tensor ycomp=y[pos];
				auto dims=ycomp.dimensions;
				auto tmpxl=xbasis[1].get_component(pos);
				auto tmpxr=xbasis[0].get_component(pos);
				if(pos==0){
						ycomp.resize_mode(2,dims[2]*2,0);

						tmpxl.resize_mode(2,dims[2]*2,dims[2]);
						ycomp+=tmpxl;
				}else if(pos==d-1){
						ycomp.resize_mode(0,dims[0]*2,dims[0]);

						tmpxr.resize_mode(0,dims[0]*2,0);
						ycomp+=tmpxr;
				}else{
						ycomp.resize_mode(2,dims[2]*2,0);

						tmpxl.resize_mode(2,dims[2]*2,dims[2]);
						ycomp+=tmpxl;
						ycomp.resize_mode(0,dims[0]*2,dims[0]);
						tmpxr.resize_mode(2,dims[2]*2,0);
						tmpxr.resize_mode(0,dims[0]*2,0);
						ycomp+=tmpxr;

				}
				Y.set_component(pos,ycomp);
			}
			return Y;
	  }

	  std::vector<Tensor> localProduct(const TTOperator& A, const TTOperator& F, value_t lambda, bool proj = true){
	    time_t begin_time;
			std::vector<Tensor> locY;
			rightAStack.clear();
			rightxStack.clear();

	    leftAStack = Tensor::ones({1,1,1,1});
			rightAStack.emplace_back(Tensor::ones({1,1,1,1}));
			leftxStack = Tensor::ones({1,1,1});
			rightxStack.emplace_back(Tensor::ones({1,1,1}));
			begin_time = time (NULL);
			for (size_t pos=d-1; pos>0;pos--){
	    	push_right_stack(pos,A,F);
	    	push_right_stack(pos,F);
			}
			begin_time = time (NULL);
			for (size_t corePosition=0;corePosition<d;corePosition++){
	      Tensor rhs1,rhs2;
				const Tensor &xi = xbasis[0].get_component(corePosition);
				const Tensor &Ai = A.get_component(corePosition);
				const Tensor &Fi = F.get_component(corePosition);

				Index i1,i2,i3,A1,A2,A3,A4,k1,k2,j1,j2,j3,j4,j5,j6;
				rhs1(i1, i2, i3) = leftAStack(i1,A1,A2, k1) * Fi(A1,i2,j1,A3) * Ai(A2,j1,j2,A4) *xi(k1, j2, k2) * rightAStack.back()(i3,A3,A4, k2);
				rhs2(i1, i2, i3) = leftxStack(i1,A1, k1) * Fi(A1,i2,j1,A2) *xi(k1, j1, k2) * rightxStack.back()(i3,A2, k2);

				locY.emplace_back(rhs1-lambda*rhs2);

				if (corePosition+1 < d) {
					push_left(corePosition,A,F);
					push_left(corePosition,F);
					rightAStack.pop_back();
					rightxStack.pop_back();
	      }
			}
			Index i,j;
			begin_time = time (NULL);
			if (proj)
				projection(locY);
			rightAStack.clear();
			rightxStack.clear();

			return locY;
		}


	  std::vector<Tensor> localProductSymmetric(const TTOperator& A, const TTOperator& F, value_t lambda){
	    time_t begin_time;
			std::vector<Tensor> locY;
			rightAStack.clear();
			rightxStack.clear();

	    leftAStack = Tensor::ones({1,1,1});
			rightAStack.emplace_back(Tensor::ones({1,1,1}));
			leftxStack = Tensor::ones({1,1,1});
			rightxStack.emplace_back(Tensor::ones({1,1,1}));
			begin_time = time (NULL);
			for (size_t pos=d-1; pos>0;pos--){
	    	push_right_stack_sym(pos,A);
	    	push_right_stack(pos,F);
			}
			begin_time = time (NULL);
			for (size_t corePosition=0;corePosition<d;corePosition++){
	      Tensor rhs1,rhs2;
				const Tensor &xi = xbasis[0].get_component(corePosition);
				const Tensor &Ai = A.get_component(corePosition);
				const Tensor &Fi = F.get_component(corePosition);

				Index i1,i2,i3,A1,A2,A3,A4,A5,A6,k1,k2,j1,j2,j3,j4,j5,j6;
				rhs1(i1, i2, i3) = leftAStack(i1,A1, k1)  * Ai(A1,i2,j1,A2) *xi(k1, j1, k2) * rightAStack.back()(i3,A2, k2);
				rhs2(i1, i2, i3) = leftxStack(i1,A1, k1)  * Fi(A1,i2,j1,A2) *xi(k1, j1, k2) * rightxStack.back()(i3,A2, k2);

				locY.emplace_back(rhs1-lambda*rhs2);

				if (corePosition+1 < d) {
					push_left_sym(corePosition,A);
					push_left(corePosition,F);
					rightAStack.pop_back();
					rightxStack.pop_back();
	      }
			}
			Index i,j;
			begin_time = time (NULL);
			projection(locY);
			rightAStack.clear();
			rightxStack.clear();

			return locY;
		}





	  std::vector<Tensor> localProduct(const TTTensor y, const TTOperator& F, bool proj = true){
	 	    time_t begin_time;
	 			std::vector<Tensor> locY;
	 			rightAStack.clear();

	 	    leftAStack = Tensor::ones({1,1,1});
	 			rightAStack.emplace_back(Tensor::ones({1,1,1}));
	 			begin_time = time (NULL);
	 			for (size_t pos=d-1; pos>0;pos--){
	 	    	push_right_stack(pos,y,F);
	 			}
	 			begin_time = time (NULL);
	 			for (size_t corePosition=0;corePosition<d;corePosition++){
	 	      Tensor rhs;
	 				const Tensor &yi = y.get_component(corePosition);
	 				const Tensor &Fi = F.get_component(corePosition);


	 				Index i1,i2,i3,A1,A2,A3,A4,k1,k2,j1,j2,j3,j4,j5,j6;
	 				rhs(i1, i2, i3) = leftAStack(i1,A1, k1) * Fi(A1,i2,j1,A2)  *yi(k1, j1, k2) * rightAStack.back()(i3,A2, k2);


	 				locY.emplace_back(rhs);

	 				if (corePosition+1 < d) {
	 					push_left(corePosition,y,F);
	 					rightAStack.pop_back();
	 	      }
	 			}
	 			Index i,j;
	 			begin_time = time (NULL);
	 			if(proj)
	 				projection(locY);
	 			rightAStack.clear();

	 			return locY;
	 		}

	  std::vector<Tensor> localProduct(const TTTensor y, bool proj = true){
		time_t begin_time;
		std::vector<Tensor> locY;
		rightAStack.clear();

		leftAStack = Tensor::ones({1,1});
		rightAStack.emplace_back(Tensor::ones({1,1}));
		begin_time = time (NULL);
		for (size_t pos=d-1; pos>0;pos--){
			push_right_stack(pos,y);
		}
		begin_time = time (NULL);
		for (size_t corePosition=0;corePosition<d;corePosition++){
			Tensor rhs;
			const Tensor &yi = y.get_component(corePosition);

			Index i1,i2,i3,A1,A2,A3,A4,k1,k2,j1,j2,j3,j4,j5,j6;
			rhs(i1, i2, i3) = leftAStack(i1, k1)  *yi(k1, i2, k2) * rightAStack.back()(i3, k2);


			locY.emplace_back(rhs);

			if (corePosition+1 < d) {
				push_left(corePosition,y);
				rightAStack.pop_back();
			}
		}
		Index i,j;
		begin_time = time (NULL);
		if(proj)
			projection(locY);
		rightAStack.clear();

		return locY;
	}

private:
	//projection onto fixed rank  tangentialspace
	void projection(std::vector<Tensor>& y){
		for (size_t pos=0;pos<d-1;pos++){
			Tensor xi=xbasis[1].get_component(pos);
			Tensor ycomp=y[pos];
			Index i1,i2,i3,     j1,j2,j3,       q;
			Tensor tmp;
			tmp(i1,i2,i3)=xi(i1,i2,q)*xi(j1,j2,q)*ycomp(j1,j2,i3);
			ycomp-=tmp;
			y[pos]=ycomp;
		}
	}


	void push_left(const size_t _position,const TTOperator& A,const TTOperator& F) {
		Index i1, i2, i3,i4,i5, j1 , j2, j3, j4, j5, k1, k2,k3,k4;
		const Tensor &xi = xbasis[1].get_component(_position);
		const Tensor &yi = xbasis[0].get_component(_position);
		const Tensor &Ai = A.get_component(_position);
		const Tensor &Fi = F.get_component(_position);

		leftAStack(i1, i2, i3, i4) = leftAStack(j1, j2, j3,j4)
				*xi(j1, k1, i1)*Fi(j2,k1,k2,i2)*Ai(j3, k2, k3, i3)*yi(j4, k3, i4);

	}

	void push_left_sym(const size_t _position,const TTOperator& A) {
		Index i1, i2, i3,i4,i5, j1 , j2, j3, j4, j5, k1, k2,k3,k4;
		const Tensor &xi = xbasis[1].get_component(_position);
		const Tensor &yi = xbasis[0].get_component(_position);
		const Tensor &Ai = A.get_component(_position);

		leftAStack(i1, i2, i3) = leftAStack(j1, j2, j3)
				*xi(j1, k1, i1)*Ai(j2, k1, k2, i2) * yi(j3, k2, i3);

	}

	void push_left(const size_t _position,const TTTensor y, const TTOperator& F) {
		Index i1, i2, i3,i4,i5, j1 , j2, j3, j4, j5, k1, k2,k3,k4;
		const Tensor &xi = xbasis[1].get_component(_position);
		const Tensor &yi = y.get_component(_position);
		const Tensor &Fi = F.get_component(_position);

		leftAStack(i1, i2, i3) = leftAStack(j1, j2, j3)
				*xi(j1, k1, i1)*Fi(j2,k1,k2,i2)*yi(j3, k2, i3);
	}

	void push_left(const size_t _position,const TTTensor y) {
		Index i1, i2, i3,i4,i5, j1 , j2, j3, j4, j5, k1, k2,k3,k4;
		const Tensor &xi = xbasis[1].get_component(_position);
		const Tensor &yi = y.get_component(_position);

		leftAStack(i1, i2) = leftAStack(j1, j2)
				*xi(j1, k1, i1)*yi(j2,k1,i2);
	}

	void push_left(const size_t _position,const TTOperator& F) {
		Index i1, i2, i3, j1 , j2, j3, k1, k2;
		const Tensor &xi = xbasis[1].get_component(_position);
		const Tensor &yi = xbasis[0].get_component(_position);
		const Tensor &Fi = F.get_component(_position);

		leftxStack(i1, i2, i3) = leftxStack(j1, j2, j3)
				*xi(j1, k1, i1)*Fi(j2, k1, k2, i2)*yi(j3, k2, i3);
	}

	void push_right_stack(const size_t _position,const TTOperator& A,const TTOperator& F) {
    Index i1, i2, i3,i4,i5, j1 , j2, j3, j4, j5, k1, k2,k3,k4;
		const Tensor &xi = xbasis[0].get_component(_position);
		const Tensor &Ai = A.get_component(_position);
		const Tensor &Fi = F.get_component(_position);

		Tensor tmpA;
		tmpA(i1, i2, i3, i4) = xi(i1, k1, j1)*Fi(i2,k1,k2,j2)*Ai(i3, k2, k3, j3)*xi(i4, k3, j4)
				*rightAStack.back()(j1, j2, j3,j4);
		rightAStack.emplace_back(std::move(tmpA));
	}

	void push_right_stack_sym(const size_t _position,const TTOperator& A) {
    Index i1, i2, i3,i4,i5, j1 , j2, j3, j4, j5, k1, k2,k3,k4;
		const Tensor &xi = xbasis[0].get_component(_position);
		const Tensor &Ai = A.get_component(_position);

		Tensor tmpA;
		tmpA(i1, i2, i3) = xi(i1, k1, j1)*Ai(i2, k1, k2, j2)*xi(i3, k2, j3)
				*rightAStack.back()(j1, j2, j3);
		rightAStack.emplace_back(std::move(tmpA));
	}


	void push_right_stack(const size_t _position,const TTTensor y) {
    Index i1, i2, i3,i4,i5, j1 , j2, j3, j4, j5, k1, k2,k3,k4;
		const Tensor &xi = xbasis[0].get_component(_position);
		const Tensor &yi = y.get_component(_position);
		Tensor tmpA;
		tmpA(i1, i2) = xi(i1, k1, j1)*yi(i2, k1, j2)
				*rightAStack.back()(j1, j2);
		rightAStack.emplace_back(std::move(tmpA));
	}

	void push_right_stack(const size_t _position,const TTTensor y, const TTOperator& F) {
    Index i1, i2, i3,i4,i5, j1 , j2, j3, j4, j5, k1, k2,k3,k4;
		const Tensor &xi = xbasis[0].get_component(_position);
		const Tensor &yi = y.get_component(_position);
		const Tensor &Fi = F.get_component(_position);

		Tensor tmpA;
		tmpA(i1, i2, i3) = xi(i1, k1, j1)*Fi(i2,k1,k2,j2)*yi(i3, k2, j3)
				*rightAStack.back()(j1, j2, j3);
		rightAStack.emplace_back(std::move(tmpA));
	}



	void push_right_stack(const size_t _position,const TTOperator& F) {
		Index i1, i2, i3, j1 , j2, j3, k1, k2;
		const Tensor &xi = xbasis[0].get_component(_position);
		const Tensor &Fi = F.get_component(_position);

		Tensor tmpA;
		tmpA(i1, i2, i3) = xi(i1, k1, j1)*Fi(i2, k1, k2, j2)*xi(i3, k2, j3)
				*rightxStack.back()(j1, j2, j3);
		rightxStack.emplace_back(std::move(tmpA));
	}


};





