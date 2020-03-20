#include <xerus.h>
#include <time.h>
#include <cmath>
#include <iostream>
#include <fstream>
#include "localProblem.cpp"
#include "eigen.cpp"
#include "gradient.cpp"
#include "vectorTransport.cpp"
#include "cgmethod.cpp"
#include "ALSres.cpp"

using namespace xerus;


class solver{
	TTTensor& x;
	const TTOperator& A;
	const TTTensor& b;
    value_t solnorm;
    Tensor xloc;
	Tensor bloc;
    Tensor resP;
    size_t currit;
	std::vector<size_t> ranks;
    
    std::ofstream& myfile;
    localProblem local1;

public:
    bool rankadaption;
    
    size_t CG_minit=15;
    size_t CG_maxit=30;
    value_t CG_error=0.0000001;

    value_t error;	
    value_t error2;	

    time_t begin_time;
    
    
    solver(const TTOperator& _A,const TTTensor& _b, TTTensor& _x,std::ofstream& file) 
	:local1(x),b(_b),  x(_x), A(_A),myfile(file)
	{ 
        currit=0;
		ranks=x.ranks();
        begin_time =time (NULL);
        solnorm=frob_norm(b);
	}
	
	void simple(){
        Index i,j,k;
        local1.update(x);
        xloc=local1.getlocalVector(x);
        local1.projection(xloc);
        bloc=local1.getlocalVector(b);
        local1.projection(bloc);
        auto tmp=local1.localProduct(xloc,A);
        resP=bloc-tmp;
        writeData();
        Tensor v=getdirection();
        xloc-=v;
        x=local1.builtTTTensor(xloc);
        x.round(ranks);
    }
    
    void writeData(){
        Index i,j;
        TTTensor res;
        res(i&0)=A(i/2,j/2)*x(j&0)-b(i&0);
        myfile<<std::setprecision(12)<<frob_norm(res)/solnorm<<std::setw(12)<<"\t"<<frob_norm(resP)/solnorm<<std::setw(12)<<"\t"<<currit<<std::setw(12)<<"\t"<<(time (NULL)-begin_time)<<std::endl;
        currit++;
    }
	
	
	Tensor getdirection(){
        Tensor rhs=(-1)*resP;

        Tensor direction=rhs;
        local1.cg_method3(A,rhs,direction,CG_maxit,CG_minit,CG_error);
        return direction;
    }
	

};
