#include <xerus.h>

using namespace xerus;
void maxeigen(TTOperator& A,TTTensor& x,size_t maxit,value_t tol){
    std::vector<size_t> ranks=x.ranks();
    Index i,j;
	size_t it=0;
	value_t error=100;
    while ((it<maxit)&&(error > tol)){
		it++;
		TTTensor y;
		y(i&0)=A(i/2,j/2)*x(j&0);
		//y.round(ranks);
		auto lambda=Tensor::ones({1});
		lambda(i)=lambda(i)*y(j&0)*x(j&0);
		error= frob_norm((y-lambda[0]*x)/lambda[0]);

		//std::cout<<it<<"\t"<<error<<"\t"<<lambda[0]<<std::endl;

		x=y;
        x.round(ranks);
        x=x/frob_norm(x);

	}    
    //std::cout<<"eigenvalue iterations "<<it<<std::endl;
    
}
void maxeigen(Tensor& A,Tensor& x,size_t maxit,value_t tol){
	Index i,j;
	size_t it=0;
	value_t error=100;
	while ((it<maxit)&&(error > tol)){
		it++;
		Tensor y;
		y(i&0)=A(i/2,j/2)*x(j&0);
		
		auto lambda=Tensor::ones({1});
		lambda(i)=lambda(i)*y(j&0)*x(j&0);
		error= frob_norm((y-lambda[0]*x)/lambda[0]);

		//std::cout<<it<<"\t"<<error<<"\t"<<lambda[0]<<std::endl;

		x=y/frob_norm(y);

	}
	//std::cout<<"eigenvalue iterations "<<it<<std::endl;


}

void targeteigen(Tensor& A,Tensor& x,size_t maxit,value_t tol, double target){
	Index i,j;
	size_t it=0;
	value_t error=100;
    x+=0.00000001*Tensor::random(x.dimensions);
	while ((it<maxit)&&(error > tol)){
		it++;
		Tensor y=x;
		Tensor Op=A-target*Tensor::identity(A.dimensions);		
		
		xerus::solve(y,Op,x);
		//minres(Op,x,y,1000,0.0000000000000001);
		Tensor z;
        if((frob_norm(y)==0)||(frob_norm(y)>100000000000000000)) return;
		y/=frob_norm(y);
		z(i&0)=A(i/2,j/2)*y(j&0);
		auto lambda=Tensor::ones({1});
		lambda(i)=lambda(i)*y(j&0)*z(j&0);
		error= frob_norm((z-lambda[0]*y)/lambda[0]);

		//std::cout<<it<<"\t"<<error<<"\t"<<lambda[0]<<std::endl;

		x=y;
        x/=frob_norm(x);

	}
//	std::cout<<"eigenvalue iterations "<<it<<"  error "<<error<<std::endl;


}

void mineigen(Tensor& A,Tensor& x,size_t maxit,value_t tol){
	targeteigen(A,x,maxit,tol, 0);
}



