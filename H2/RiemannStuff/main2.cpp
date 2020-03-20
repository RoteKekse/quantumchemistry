#include <xerus.h>

#include <iostream>
#include <fstream>
#include "laplace.cpp"
#include "RiemannianLinear.cpp"
#include "ALSEV.cpp"


int main(){
    std::ofstream myfile;
    myfile.open("testLinear.dat");
    
    

	size_t order=20;
	size_t dimension=2;
	std::vector<size_t> dims1=std::vector<size_t>(order, dimension);
	std::vector<size_t> dims2=std::vector<size_t>(order*2, dimension);
	
    TTTensor  ttx = TTTensor::random(dims1,{ 10 });//{2, 4, 8, 8, 12, 13, 12, 8, 12, 13, 12, 8, 8, 4, 2 });//{ 2, 4, 8, 8, 9, 9, 12, 12, 9, 9, 9, 7, 8, 4, 2 });//{ 2, 4, 5, 4, 3, 2, 3, 4, 5, 4, 3, 2, 3, 2, 3, 2, 3, 2, 2 });//
    //{ 2, 4, 6, 5, 4, 3, 4, 5, 6, 5, 4, 3, 4, 2, 3, 2, 3, 2, 2 });
    using xerus::misc::operator<<;
    std::cout<<ttx.ranks()<<std::endl;
    
    TTOperator ttA;
    //ttA=0.000*TTOperator::random(dims2,{1});ttA(i/2,j/2)=ttA(i/2,k/2)*ttA(j/2,k/2);
    //ttA+=sym_laplace_operator1(dims1);
    
    std::ifstream read("hamiltonian20_full.ttoperator");
    misc::stream_reader(read,ttA,xerus::misc::FileFormat::BINARY);
    std::cout<<ttA.ranks()<<std::endl<<std::endl;
    ttA+=3*TTOperator::identity(ttA.dimensions);
    Index i,j;
    TTTensor ttb;
    ttb(i&0)=ttA(i/2,j/2)*ttx(j&0);
    ttb.round({20});
    size_t it=0;
    
    //doALS(ttx, ttA, myfileALS,-15  ,20);
    auto tty=TTTensor::random(dims1,{10}); 
    //doALS(tty, ttA, myfileALS,-110.9 ,20);
    std::cout<<"now solve"<<std::endl;
    solver help(ttA,ttb,tty,myfile);
    while(it<2000){
        it++;
        help.simple();
        std::cout<<frob_norm(ttx-tty)/frob_norm(ttx)<<std::endl;
    }

    
    
    
    
    
    
}
