#include <xerus.h>
using namespace xerus;
using xerus::misc::operator<<;

#pragma once

void write_to_disc(std::string name, TTOperator &op){
	std::ofstream write(name.c_str() );
	xerus::misc::stream_writer(write,op,xerus::misc::FileFormat::BINARY);
	write.close();
}

void write_to_disc(std::string name, TTTensor &x){
	std::ofstream write(name.c_str() );
	xerus::misc::stream_writer(write,x,xerus::misc::FileFormat::BINARY);
	write.close();
}


void read_from_disc(std::string name, TTOperator &op){
	std::ifstream read(name.c_str() );
	xerus::misc::stream_reader(read,op,xerus::misc::FileFormat::BINARY);
	read.close();

}

void read_from_disc(std::string name, TTTensor &x){
	std::ifstream read(name.c_str() );
	xerus::misc::stream_reader(read,x,xerus::misc::FileFormat::BINARY);
	read.close();

}


value_t contract_TT(const TTOperator& A, const TTTensor& x, const TTTensor& y){
	Tensor stack = Tensor::ones({1,1,1});
	size_t d = A.order()/2;
	Index i1,i2,i3,j1,j2,j3,k1,k2;
	for (size_t i = 0; i < d; ++i){
		auto Ai = A.get_component(i);
		auto xi = x.get_component(i);
		auto yi = y.get_component(i);

		stack(i1,i2,i3) = stack(j1,j2,j3) * xi(j1,k1,i1) * Ai(j2,k1,k2,i2) * yi(j3,k2,i3);
	}
	return stack[0,0,0];
}


value_t contract_TT2(const TTOperator& A,const TTOperator& B, const TTTensor& x, const TTTensor& y){
	Tensor stack = Tensor::ones({1,1,1,1});
	size_t d = A.order()/2;
	Index i1,i2,i3,i4,j1,j2,j3,j4,k1,k2,k3;
	for (size_t i = 0; i < d; ++i){
		XERUS_LOG(info,i);
		auto Ai = A.get_component(i);
		auto Bi = B.get_component(i);
		auto xi = x.get_component(i);
		auto yi = y.get_component(i);

		stack(i1,i2,i3,i4) = stack(j1,j2,j3,j4) * xi(j1,k1,i1) * Ai(j2,k1,k2,i2) * Bi(j3,k2,k3,i3) * yi(j4,k3,i4);
	}
	return stack[0,0,0,0];
}
