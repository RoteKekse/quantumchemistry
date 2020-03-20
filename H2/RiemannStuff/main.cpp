#include <xerus.h>

#include <iostream>
#include <fstream>
#include "laplace.cpp"
#include "solver.cpp"
#include "ALSEV.cpp"


int main(){
    //hier speichere ich die ergebnisse

    std::ofstream myfile;
    myfile.open("test.dat");

    Index i,j,k;
	size_t order=20;
	size_t dimension=2;
	size_t ranks = 20;
	size_t start_ranks = 5;
	std::vector<size_t> dims1=std::vector<size_t>(order, dimension);
	std::vector<size_t> dims2=std::vector<size_t>(order*2, dimension);
	
    //hier startraenge angeben
    
    TTTensor  ttx = TTTensor::random(dims1,{ start_ranks });//{2, 4, 8, 8, 12, 13, 12, 8, 12, 13, 12, 8, 8, 4, 2 });//{ 2, 4, 8, 8, 9, 9, 12, 12, 9, 9, 9, 7, 8, 4, 2 });//{ 2, 4, 5, 4, 3, 2, 3, 4, 5, 4, 3, 2, 3, 2, 3, 2, 3, 2, 2 });//
    //{ 2, 4, 6, 5, 4, 3, 4, 5, 6, 5, 4, 3, 4, 2, 3, 2, 3, 2, 2 });
    using xerus::misc::operator<<;
//    std::cout<<ttx.ranks()<<std::endl;
//    std::ifstream read("../eigenvector_H2O_50_-85.042634.ttoperator");
//		misc::stream_reader(read,ttx,xerus::misc::FileFormat::BINARY);
//		std::cout<<ttx.ranks()<<std::endl<<std::endl;
    TTOperator ttA;
    //ttA=0.000*TTOperator::random(dims2,{1});ttA(i/2,j/2)=ttA(i/2,k/2)*ttA(j/2,k/2);
    //ttA+=sym_laplace_operator1(dims1);
    //hier operatorladen
    std::ifstream read2("../hamiltonian20_full.ttoperator");
    misc::stream_reader(read2,ttA,xerus::misc::FileFormat::BINARY);
    std::cout<<ttA.ranks()<<std::endl<<std::endl;
   
    size_t it=0;
    
    //doALS(ttx, ttA, myfileALS,-15  ,20);
    auto tty=ttx; 
    //doALS(tty, ttA, myfileALS,-110.9 ,20);
    std::cout<<"now solve"<<std::endl;
    
    //methode faengt hier an, die erste zahl sollte etwas kleiner, das switchlambda etwas groesser als der kleinste eigenwert sein
    
    solver help(ttA,ttx,-1.9 ,myfile);
    help.switchlambda=-1.7;
    //help.targetadaption=true;
    help.loc=false;
    //epsilon und maxranks festlegen
    help.epsilon=10e-18;
    help.maxranks=std::vector<size_t>(order-1,ranks);
    
    //2000 iterationen
    while(it<2000){
        it++;
   //das ist drinne falls switchlambda zu klein gewaehlt ist
        if (it>20)
            help.targetadaption=true;
       
        help.simple();
    }

}
