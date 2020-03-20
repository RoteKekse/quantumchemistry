#include <xerus.h>
#include "projection.cpp"


using namespace xerus;

class gradient{
    TTTensor& x;

	const TTOperator& A;
    
public:
    TTTensor Ax;
    
    gradient(TTTensor& _x, const TTOperator& _A)
	:x(_x),A(_A){

	}
	TTTensor return_gradient(){
        x/=frob_norm(x);
        Index i,j;
        Ax(i&0)=A(i/2,j/2)*x(j&0);
        projection help(x,Ax);
        Ax=help.b_p;
        return Ax;
    }
	
    
};
