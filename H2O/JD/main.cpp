#include <xerus.h>

#include <iostream>
#include <fstream>
#include "solver.cpp"
#include "quantOp.cpp"
int main(){
    //hier speichere ich die ergebnisse

    std::ofstream myfile;
    myfile.open("test10er.dat");
/*
    std::ofstream myfile2;
    myfile2.open("test20s.dat");
    
    std::ofstream myfile3;
    myfile3.open("test10.dat");
    
    std::ofstream myfile4;
    myfile4.open("test10s.dat");
  
    std::ofstream myfile5;
    myfile5.open("test_gradient2.dat");
 */  

    Index i,j,k;
	size_t order=10;
	size_t dimension=30;
	std::vector<size_t> dims1=std::vector<size_t>(order, dimension);
	std::vector<size_t> dims2=std::vector<size_t>(order*2, dimension);
    //hier startraenge angeben
    
    TTTensor  ttx =TTTensor::random(dims1,50);//{ 2, 4, 8, 16, 32, 54, 76, 85, 97, 99, 87, 62, 61, 57, 87, 71, 62, 36, 44, 33, 26, 16, 8, 4, 2 });//{ 2, 4, 8, 16, 32, 61, 97, 126, 160, 198, 214, 213, 218, 147, 164, 138, 118, 88, 99, 57, 31, 16, 8, 4, 2 });//{2, 4, 8, 8, 12, 13, 12, 8, 12, 13, 12, 8, 8, 4, 2 });//{ 2, 4, 8, 8, 9, 9, 12, 12, 9, 9, 9, 7, 8, 4, 2 });//{ 2, 4, 5, 4, 3, 2, 3, 4, 5, 4, 3, 2, 3, 2, 3, 2, 3, 2, 2 });//
    //{ 2, 4, 6, 5, 4, 3, 4, 5, 6, 5, 4, 3, 4, 2, 3, 2, 3, 2, 2 });
    using xerus::misc::operator<<;
    
    
    TTOperator ttA;
/*
      std::ifstream readttx("CH2HF.tttensor");
        misc::stream_reader(readttx,ttx,xerus::misc::FileFormat::BINARY);
    ttx+=1e-4*TTTensor::random(dims1,1);
    ttx/=frob_norm(ttx);
    for(size_t it=0;it<order;it++){
            std::cout<<ttx.get_component(it).to_string()<<std::endl;
    }
  */  
//  std::cout<<number[0]<<" electrons"<<std::endl;
//    std::ofstream write("CH2HF.tttensor");
//    misc::stream_writer(write,ttx,xerus::misc::FileFormat::BINARY);    
      
//      

    std::ifstream read("henon_heiles_10_30_011.ttoperator");
    misc::stream_reader(read,ttA,xerus::misc::FileFormat::BINARY);
    std::cout<<ttA.ranks()<<std::endl<<std::endl;
    std::cout<<ttx.ranks()<<std::endl<<std::endl;
    
    TTOperator P=particleNumber(order);

    auto tty=ttx;

    std::cout<<"now solve"<<std::endl;
    
    //methode faengt hier an, die erste zahl sollte etwas kleiner, das switchlambda etwas groesser als der kleinste eigenwert sein
    size_t it=0;

    
    solver help1(ttA,ttx,5,myfile);
    //epsilon und maxranks festlegen
    help1.rounderror=1e-12;
    help1.maxranks={100,100,100,100,100,100,100,100,100};
  //  help1.maxranks={2, 4, 8, 16, 32, 64, 128, 170, 200, 250, 260, 260, 260, 200, 200, 190, 180, 170, 128, 64, 32, 16, 8, 4, 2 };
    help1.switchlambda=5.2;
    //help.calcresidual=true;
    
    //2000 iterationen
    
    help1.CG_maxit=20;
    help1.CG_error=1e-6;
    //less matrixmultiplic
    while(it<500){
        it++;
            help1.simple();
            //help1.subspacetransportsolve(1);
            //help1.tangentspacesimple(5);
        Tensor number;
        //number()=ttx(i&0)*P(i/2,j/2)*ttx(j&0);
        //std::cout<<number[0]<<" electrons"<<std::endl;
        std::ofstream write("henon_heiles_phi.tttensor");
        misc::stream_writer(write,ttx,xerus::misc::FileFormat::BINARY);
    }
    
    help1.writeData(true);
    
    
    /*
    ttx=tty;
    solver help(ttA,ttx,-10.8,myfile2);
    //epsilon und maxranks festlegen
    help.rounderror=1e-12;
    help.maxranks={2, 4, 8, 16, 32, 64, 110, 140, 170, 210, 230, 230, 235, 170, 180, 150, 140, 110, 120, 64, 32, 16, 8, 4, 2 };//std::vector<size_t>(order-1,220);
    help.switchlambda=-10.7;
    //help.calcresidual=true;
    help.CG_maxit=20;
    help.CG_error=1e-6;
    //2000 iterationen
    
    it=0;
    //simple
    while(it<4){
        it++;
            //help.simple();
            help.subspacetransportsolve(5);
            //help.tangentsubspacespacesolve(5);
        Tensor number;
        number()=ttx(i&0)*P(i/2,j/2)*ttx(j&0);
        std::cout<<number[0]<<" electrons"<<std::endl;
        std::ofstream write("CH2Ray.tttensor");
        misc::stream_writer(write,ttx,xerus::misc::FileFormat::BINARY);
    }
    //help.writeData(true);
    
    
    
   ttx=tty;
    
    
    
    
    solver help2(ttA,ttx,-10.8,myfile3);
    //epsilon und maxranks festlegen
    help2.rounderror=1e-12;
    help2.maxranks={2, 4, 8, 16, 32, 64, 110, 140, 170, 210, 230, 230, 235, 170, 180, 150, 140, 110, 120, 64, 32, 16, 8, 4, 2 };
    help2.switchlambda=-10.77;
    //help.calcresidual=true;
    it=0;
    //2000 iterationen  3.90362941971e-09		0.0001862       2.41082176444e-09	
    
    
    help2.CG_maxit=10;
    help2.CG_error=1e-6;
    //subspace
    while(it<8){
        it++;
            //help.simple();
            help2.subspacetransportsolve(5);
            //help2.tangentsubspacespacesolve(5);
        Tensor number;
        number()=ttx(i&0)*P(i/2,j/2)*ttx(j&0);
        std::cout<<number[0]<<" electrons"<<std::endl;
        std::ofstream write("CH2JD.tttensor");
        misc::stream_writer(write,ttx,xerus::misc::FileFormat::BINARY);
    }
    //help2.writeData(true);
    
    
    
   ttx=tty;
    
    solver help3(ttA,ttx,-10.8,myfile4);
    //epsilon und maxranks festlegen
    help3.rounderror=1e-12;
    help3.maxranks={2, 4, 8, 16, 32, 64, 110, 140, 170, 210, 230, 230, 235, 170, 180, 150, 140, 110, 120, 64, 32, 16, 8, 4, 2 };
    help3.switchlambda=-10.77;
    //help.calcresidual=true;
    it=0;
    //2000 iterationen
    help3.CG_maxit=10;
    help3.CG_error=1e-6;
        //subspace
    while(it<40){
        it++;
            help3.simple();
            //help3.subspacetransportsolve(5);
            //help2.tangentsubspacespacesolve(5);
        Tensor number;
        number()=ttx(i&0)*P(i/2,j/2)*ttx(j&0);
        std::cout<<number[0]<<" electrons"<<std::endl;
        std::ofstream write("CH2JDlite.tttensor");
        misc::stream_writer(write,ttx,xerus::misc::FileFormat::BINARY);
    }
    help3.writeData(true);
    
    
    
    ttx=tty;
    
    solver help4(ttA,ttx,-10.8,myfile5);
    //epsilon und maxranks festlegen
    help4.rounderror=1e-12;
    help4.maxranks={2, 4, 8, 16, 32, 64, 110, 140, 170, 210, 230, 230, 235, 170, 180, 150, 140, 110, 120, 64, 32, 16, 8, 4, 2 };
    help4.switchlambda=-10.77;
    //help.calcresidual=true;
     it=0;
    //2000 iterationen
    help4.CG_maxit=0;
    
    
    //gradient
    while(it<1000){
        it++;
            help4.simple();
            //help4.subspacetransportsolve(5);
            //help.tangentsubspacespacesolve(5);
        Tensor number;
        number()=ttx(i&0)*P(i/2,j/2)*ttx(j&0);
        std::cout<<number[0]<<" electrons"<<std::endl;
        std::ofstream write("CH2grad.tttensor");
        misc::stream_writer(write,ttx,xerus::misc::FileFormat::BINARY);
    }
    help4.writeData(true);
*/
    
    
    
    
}
