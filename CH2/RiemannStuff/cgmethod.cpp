#include <xerus.h>


using namespace xerus;

void cg_method(const Tensor& A,const Tensor& P, const Tensor& rhs, Tensor& x, size_t maxit, size_t  minit, value_t minerror){
    value_t error=100;
    size_t it=0;
    Tensor r,d,z;
    Index i,j;
    double alpha,beta,rhsnorm;
    rhsnorm=frob_norm(rhs);
    r(i&0)=rhs(i&0)-A(i/2,j/2)*x(j&0);
    d=r;
    size_t dims=1;
    for (size_t n=0;n<x.dimensions.size();n++){
        dims*=x.dimensions[n];
    }
    while((it<dims-1)&&(((error>minerror*minerror)&&(it<maxit))||(it<minit))){
        z(i&0)=A(i/2,j/2)*d(j&0);
        Tensor tmp1,tmp2;
        tmp1()=r(i&0)*r(i&0);
        tmp2()=d(i&0)*z(i&0);
        alpha=tmp1[0]/tmp2[0];
        x+=alpha*d;
        x(i&0)=P(i/2,j/2)*x(j&0);
        r-=alpha*z;
        tmp2()=r(i&0)*r(i&0);
        beta=tmp2[0]/tmp1[0];
        d=r+beta*d;
        
        d(i&0)=P(i/2,j/2)*d(j&0);
        error=tmp2[0];//(rhsnorm*rhsnorm);
        it++;
    }
    //std::cout<<"error^2="<<error<<" iterations: "<<it<<std::endl;
}


void pcg_method(const Tensor& A,const Tensor& M,const Tensor& P, const Tensor& rhs, Tensor& x, size_t maxit, size_t  minit, value_t minerror){
    size_t dims=1;
    for (size_t n=0;n<x.dimensions.size();n++){
        dims*=x.dimensions[n];
    }
    
    
    value_t error=100;
    size_t it=0;
    Tensor r,h,d,z;
    Index i,j;
    double alpha,beta,rhsnorm;
    rhsnorm=frob_norm(rhs);
    r(i&0)=rhs(i&0)-A(i/2,j/2)*x(j&0);
    xerus::solve(h,M,r);
    d=h;
    
    while((it<dims-1)&&(((error>minerror)&&(it<maxit))||(it<minit))){
        z(i&0)=A(i/2,j/2)*d(j&0);
        Tensor tmp1,tmp2;
        tmp1()=r(i&0)*h(i&0);
        tmp2()=d(i&0)*z(i&0);
        alpha=tmp1[0]/tmp2[0];
        x+=alpha*d;
        
        r-=alpha*z;
        
        xerus::solve(h,M,r);
        
        x(i&0)=P(i/2,j/2)*x(j&0);
        
        tmp2()=r(i&0)*h(i&0);
        beta=tmp2[0]/tmp1[0];
        
        d=h+beta*d;
        d(i&0)=P(i/2,j/2)*d(j&0);
        it++;
        error=frob_norm(r);
    }
    std::cout<<"error="<<error<<" iterations: "<<it<<std::endl;
    
    
    
}
