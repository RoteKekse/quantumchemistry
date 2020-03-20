#include <xerus.h>


using namespace xerus;



void add(std::vector<Tensor>& y,const std::vector<Tensor>& z){
    size_t d=y.size();
    for (size_t pos=0; pos<d;pos++){
        y[pos]+=z[pos];
    }
}

void add(std::vector<Tensor>& y,const std::vector<Tensor>& z, double alpha){
    size_t d=y.size();
    for (size_t pos=0; pos<d;pos++){
        y[pos]+=alpha*z[pos];
    }
}

void project(std::vector<Tensor>& x,const std::vector<Tensor>& Q){
    size_t d=x.size();
    for (size_t pos=0; pos<d;pos++){
        Index i;
        Tensor tmp;
        tmp()=x[pos](i&0)*Q[pos](i&0);
        x[pos]-=tmp[0]*Q[pos];
    }
}


void multiply(std::vector<Tensor>& y,double alpha){
    size_t d=y.size();
    for (size_t pos=0; pos<d;pos++){
        y[pos]*=alpha;
    }
}

double innerprod(const std::vector<Tensor>& y,const std::vector<Tensor>& z){
    double result=0;
    size_t d=y.size();
    for (size_t pos=0; pos<d;pos++){
        Index i;
        Tensor tmp;
        tmp()=y[pos](i&0)*z[pos](i&0);
        result+=tmp[0];
    }
    return result;
}
value_t frob_norm(const std::vector<Tensor>& y){
    size_t d=y.size();
    double result=0;
    
    for (size_t pos=0; pos<d;pos++){
        auto tmp= frob_norm(y[pos])*frob_norm(y[pos]);
        result+=tmp;
    }

    result=sqrt(result);
    return result;
}

