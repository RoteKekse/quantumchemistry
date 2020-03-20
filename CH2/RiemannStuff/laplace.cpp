#include <xerus.h>
using namespace xerus;


TTOperator sym_laplace_operator(std::vector<size_t> dimensions){
	Index i,j,k;
	std::vector<size_t> dims2=dimensions;
	for (size_t pos=0; pos<dimensions.size();pos++){
		dims2.emplace_back(dimensions[pos]);	
	}
		
	TTOperator I= TTOperator::identity(dims2);
	TTOperator L= TTOperator(dims2);
	for (size_t pos=0; pos<dimensions.size();pos++){
		auto A = TTOperator(dims2);
		for (size_t pos1=0; pos1<dimensions.size();pos1++){
			Tensor comp;//=component;			
			if(pos1==pos){
				comp=Tensor::random({dimensions[pos1],dimensions[pos1]});
				comp(i,j)=comp(i,k)*comp(j,k);
			}else{
				comp=Tensor::identity({dimensions[pos1],dimensions[pos1]});
			}
			comp.reinterpret_dimensions({1,dimensions[pos1],dimensions[pos1],1});
			A.set_component(pos1,comp);
		} 
		
		L+=A;
		auto test=L;
		L.round({2});
		using xerus::misc::operator<<;
		std::cout<<frob_norm(L-test)<<std::endl;
		std::cout<<L.ranks()<<std::endl;
	} 
	return L;
}

TTOperator sym_laplace_operator1(std::vector<size_t> dimensions){
    Index i,j,k;
    std::vector<size_t> dims2=dimensions;
	for (size_t pos=0; pos<dimensions.size();pos++){
		dims2.emplace_back(dimensions[pos]);	
	}
	TTOperator L= TTOperator::random(dims2,{2});
    for (size_t pos1=0; pos1<dimensions.size();pos1++){
        Tensor comp;//=component;
        comp=Tensor::random({dimensions[pos1],dimensions[pos1]});
        comp(i,j)=comp(i,k)*comp(j,k);
        comp.reinterpret_dimensions({1,dimensions[pos1],dimensions[pos1],1});
        if(pos1==0){
            comp.resize_mode(3,2,0);
            auto I=Tensor::identity({dimensions[pos1],dimensions[pos1]});
            I.reinterpret_dimensions({1,dimensions[pos1],dimensions[pos1],1});
            I.resize_mode(3,2,1);
            comp+=I;
        }else if (pos1==dimensions.size()-1){
            comp.resize_mode(0,2,1);
            auto I=Tensor::identity({dimensions[pos1],dimensions[pos1]});
            I.reinterpret_dimensions({1,dimensions[pos1],dimensions[pos1],1});
            I.resize_mode(0,2,0);
            comp+=I;
            
        }else{
            comp.resize_mode(3,2,0);
            auto I=Tensor::identity({dimensions[pos1],dimensions[pos1]});
            I.reinterpret_dimensions({1,dimensions[pos1],dimensions[pos1],1});
            I.resize_mode(3,2,1);
            comp+=I;
            comp.resize_mode(0,2,1);
            I=Tensor::identity({dimensions[pos1],dimensions[pos1]});
            I.reinterpret_dimensions({1,dimensions[pos1],dimensions[pos1],1});
            I.resize_mode(0,2,0);
            I.resize_mode(3,2,0);
            comp+=I;
            
        }
        L.set_component(pos1,comp);
    }
    using xerus::misc::operator<<;
    std::cout<<L.ranks()<<std::endl;
    return L;
}
/*
int main(){
    Tensor T=Tensor::random({4,4});
    auto test1=sym_laplace_operator({4,4,4,4,4,4},T);
    auto test2=sym_laplace_operator1({4,4,4,4,4,4},T);
    
    std::cout<<frob_norm(test1-test2)<<std::endl;
}

*/


