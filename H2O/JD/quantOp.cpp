#include <xerus.h>

using namespace xerus;


TTTensor HartreeFock(size_t order, size_t N){
    std::vector<size_t> dims=std::vector<size_t>(order, 2);
    TTTensor result=TTTensor(dims);
    for (size_t n=N;n<order;n++){
        Tensor tmp({1,2,1});
        tmp[0]=1;
        tmp[1]=0;
        result.set_component(n,tmp);
    }
    
    for (size_t n=0;n<N;n++){
        Tensor tmp({1,2,1});
        tmp[0]=0;
        tmp[1]=1;
        result.set_component(n,tmp);
    }

   
    return result;
}


TTOperator particleNumber(size_t order){
    std::vector<size_t> dims=std::vector<size_t>(2*order, 2);
    TTOperator result=TTOperator::identity(dims);
    for(size_t n=0;n<order;n++){
        Tensor tmp;
        if(n==0){
            tmp=Tensor({1,2,2,2});
            tmp[{0,0,0,0}]=1;
            tmp[{0,1,1,0}]=1;
            tmp[{0,1,1,1}]=1;
        }else if(n==order-1){
            tmp=Tensor({2,2,2,1});
            tmp[{1,1,1,0}]=1;
            tmp[{1,0,0,0}]=1;
            tmp[{0,1,1,0}]=1;
            
        }else{
            tmp=Tensor({2,2,2,2});
            tmp[{0,0,0,0}]=1;
            tmp[{0,1,1,0}]=1;
            tmp[{0,1,1,1}]=1;
            tmp[{1,0,0,1}]=1;
            tmp[{1,1,1,1}]=1;
        }
        result.set_component(n,tmp);
   }
   return result;
    
}
