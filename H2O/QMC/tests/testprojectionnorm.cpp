#include <xerus.h>
#include "../GradientMethods/tangentialOperation.cpp"
#include "../GradientMethods/basic.cpp"
#include "../loading_tensors.cpp"

using namespace xerus;
using xerus::misc::operator<<;


int main(){
	Index i1, i2, i3, i4, j1 , j2, j3, k1, k2,r1,r2,r3,r4,n1,n2,n3,n4;
	size_t d = 48;
	size_t idx = 46;


	Tensor lstack=Tensor::ones({1,1}),rstack=Tensor::ones({1,1}),mstack,full,test;

	xerus::TTTensor phi(std::vector<size_t>(d,2));
	//read_from_disc("../data/eigenvector_H2O_48_3_-23.647510_benchmark.tttensor",phi);
	read_from_disc("../data/eigenvector_H2O_48_60_-23.824020_benchmark.tttensor",phi);
	XERUS_LOG(info,"ranks  = " << phi.ranks());

	phi.move_core(d-1);
	mstack(i1,i2,r1,n1) = phi.component(idx)(i1,i2,r2)*phi.component(idx)(r1,n1,r2);
	for(size_t i = 0; i < idx; ++i){
		lstack(i1,i2) = lstack(j1,j2) * phi.component(i)(j1,k1,i1)*phi.component(i)(j2,k1,i2);
	}

	phi.move_core(0);
	for(size_t i = d-1; i > idx; --i){
		rstack(i1,i2) =  phi.component(i)(i1,k1,j1)*phi.component(i)(i2,k1,j2) *rstack(j1,j2) ;
	}


	std::vector<size_t> ranks = phi.ranks();
	std::vector<size_t> dims = phi.dimensions;
	Tensor id = Tensor::identity({ranks[idx-1],dims[idx],ranks[idx-1],dims[idx]});

	test(i1,i2,r1,n1,i3,i4,r2,n2) = mstack(i1,i2,r1,n1)*mstack(i3,i4,r2,n2);
	XERUS_LOG(info,"Id  = " << std::pow(id.frob_norm(),2));
	XERUS_LOG(info,"Id  = " << std::pow(mstack.frob_norm(),2));
	XERUS_LOG(info,"Id  = " << std::pow(test.frob_norm(),2));


	mstack = id - mstack;
	full() = mstack(i1,i2,j1,n1)*mstack(i1,i2,j2,n1)*lstack(j1,j2)*rstack(k1,k1);

	XERUS_LOG(info,"-----------------------------------------------");
	XERUS_LOG(info,"Index  = " << idx);
	XERUS_LOG(info,"rank 1 = " << ranks[idx-1]);
	XERUS_LOG(info,"dim    = " << dims[idx]);
	XERUS_LOG(info,"rank 2 = " << ranks[idx]);
	XERUS_LOG(info,"-----------------------------------------------");
	XERUS_LOG(info,"lstack norm  = " << lstack.frob_norm());
	XERUS_LOG(info,"mstack norm  = " << mstack.frob_norm());
	XERUS_LOG(info,"rstack norm  = " << rstack.frob_norm());

	XERUS_LOG(info,"-----------------------------------------------");

	XERUS_LOG(info,"Frobenius Norm " << std::sqrt(full[0]));
	XERUS_LOG(info,"Calculated     " << std::sqrt(ranks[idx]*(dims[idx]*ranks[idx-1]-ranks[idx])));

	return 0;
}
