#include <xerus.h>
#include "basic.cpp"

using namespace xerus;



class localProblem{
    const size_t d;
    std::vector<TTTensor> xbasis;
    
    std::vector<Tensor> leftAStack;
    std::vector<Tensor> rightAStack;
    std::vector<Tensor> middleAStack;
    
    std::vector<Tensor> leftYStack;
    std::vector<Tensor> rightYStack;
    
    std::vector<Tensor> Axleft;
    std::vector<Tensor> Axright;
    
public:
    
    //projection onto fixed rank  tangentialspace
    void projection(std::vector<Tensor>& y){
        time_t begin_time = time (NULL);

        for (size_t pos=0;pos<d-1;pos++){
			//TTTensor ypos=xbasis[pos];
			Tensor xi=xbasis[1].get_component(pos);
	
			Tensor ycomp=y[pos];
 
            Index i1,i2,i3,     j1,j2,j3,       q;
            Tensor tmp;
            tmp(i1,i2,i3)=xi(i1,i2,q)*xi(j1,j2,q)*ycomp(j1,j2,i3);
            ycomp-=tmp;
            y[pos]=ycomp;
            
            
        }
        //std::cout<<"time for projection: "<<time (NULL)-begin_time<<" sekunden"<<std::endl;
    } 
    
    localProblem(TTTensor& x) 
    : d(x.degree())
    { 

            x.move_core(0,true);
            xbasis.emplace_back(x);
            x.move_core(d-1,true);
            xbasis.emplace_back(x);
        
    }
    
    void update(TTTensor& x){
        xbasis.clear();

        x.move_core(0,true);
        xbasis.emplace_back(x);
        x.move_core(d-1,true);
        xbasis.emplace_back(x);
      
        
    }
  
   
    void push_left_stack(const size_t _position,const TTTensor& y) {
		Index i1, i2, i3, j1 , j2, j3, k1, k2;
		const Tensor &xi = xbasis[1].get_component(_position);
		const Tensor &yi = y.get_component(_position);

		
		Tensor tmpy;
		tmpy(i1, i2) = leftYStack.back()(j1, j2)
				*xi(j1, k1, i1)*yi(j2, k1, i2);
		leftYStack.emplace_back(std::move(tmpy));

	}
	
	
	void push_right_stack(const size_t _position,const TTTensor& y){
        Index i1, i2, i3, j1 , j2, j3, k1, k2;
        const Tensor &xi = xbasis[0].get_component(_position);
		const Tensor &yi = y.get_component(_position);
		
		Tensor tmpy;
		tmpy(i1, i2) = xi(i1, k1, j1)*yi(i2, k1, j2)
				*rightYStack.back()(j1, j2);
		rightYStack.emplace_back(std::move(tmpy));

	}
	
	void push_left_stack(const size_t _position,const TTTensor& y,const TTOperator& A) {

        Index i1, i2, i3, j1 , j2, j3, k1, k2;
		const Tensor &xi = xbasis[1].get_component(_position);
		const Tensor &Ai = A.get_component(_position);
        const Tensor &yi = y.get_component(_position);

		Tensor tmpA;
		tmpA(i1, i2, i3) = leftYStack.back()(j1, j2, j3)
				*xi(j1, k1, i1)*Ai(j2, k1, k2, i2)*yi(j3, k2, i3);
		leftYStack.emplace_back(std::move(tmpA));

	}
	
	
	void push_right_stack(const size_t _position,const TTTensor& y,const TTOperator& A) {

        Index i1, i2, i3, j1 , j2, j3, k1, k2;
		const Tensor &xi = xbasis[0].get_component(_position);
		const Tensor &Ai = A.get_component(_position);
		const Tensor &yi = y.get_component(_position);
        

		Tensor tmpA;
		tmpA(i1, i2, i3) = xi(i1, k1, j1)*Ai(i2, k1, k2, j2)*yi(i3, k2, j3)
				*rightYStack.back()(j1, j2, j3);
		rightYStack.emplace_back(std::move(tmpA));

	}
	
    
    void push_left_stack(const size_t _position,const TTOperator& A) {
		Index i1, i2, i3, j1 , j2, j3, k1, k2;
		const Tensor &xi = xbasis[1].get_component(_position);
		const Tensor &Ai = A.get_component(_position);

		
		Tensor tmpA;
		tmpA(i1, i2, i3) = leftAStack.back()(j1, j2, j3)
				*xi(j1, k1, i1)*Ai(j2, k1, k2, i2)*xi(j3, k2, i3);
		leftAStack.emplace_back(std::move(tmpA));

	}
	
	
	void push_right_stack(const size_t _position,const TTOperator& A) {
		Index i1, i2, i3, j1 , j2, j3, k1, k2;
		const Tensor &xi = xbasis[0].get_component(_position);
		const Tensor &Ai = A.get_component(_position);
		
		Tensor tmpA;
		tmpA(i1, i2, i3) = xi(i1, k1, j1)*Ai(i2, k1, k2, j2)*xi(i3, k2, j3)
				*rightAStack.back()(j1, j2, j3);
		rightAStack.emplace_back(std::move(tmpA));

	}


    std::vector<Tensor> getlocalVector(const TTTensor& y){
        time_t begin_time = time (NULL);
        std::vector<Tensor> locY;
        
        leftYStack.clear();
		rightYStack.clear();

        leftYStack.emplace_back(Tensor::ones({1,1}));
		rightYStack.emplace_back(Tensor::ones({1,1}));
        
		for (size_t pos=d-1; pos>0;pos--){
			push_right_stack(pos,y);
		}
		
        
        for (size_t corePosition=0;corePosition<d;corePosition++){
            Tensor rhs;
			const Tensor &bi = y.get_component(corePosition);
            
            Index i1,i2,i3,k1,k2;
            rhs(i1, i2, i3) = leftYStack.back()(i1, k1) * bi(k1, i2, k2) * rightYStack.back()(i3, k2);
            
            
            locY.emplace_back(rhs);
            
            if (corePosition+1 < d) {
            
				push_left_stack(corePosition,y);
				rightYStack.pop_back();
            }
        }
        projection(locY);
        //std::cout<<"time for y components: "<<time (NULL)-begin_time<<" sekunden"<<std::endl;
        return locY;
        
    }
    
    TTTensor builtTTTensor(const std::vector<Tensor>& y){
        time_t begin_time = time (NULL);
        
        TTTensor Y(xbasis[0].dimensions);
        
        
		for (size_t pos=0;pos<d;pos++){
			
			Tensor ycomp=y[pos];
            auto dims=ycomp.dimensions;
            auto tmpxl=xbasis[1].get_component(pos);
			auto tmpxr=xbasis[0].get_component(pos);
            if(pos==0){
                ycomp.resize_mode(2,dims[2]*2,0);
                
                tmpxl.resize_mode(2,dims[2]*2,dims[2]);
                ycomp+=tmpxl;
            }else if(pos==d-1){
                ycomp.resize_mode(0,dims[0]*2,dims[0]);
                
                tmpxr.resize_mode(0,dims[0]*2,0);
                ycomp+=tmpxr;
            }else{
                ycomp.resize_mode(2,dims[2]*2,0);
                
                tmpxl.resize_mode(2,dims[2]*2,dims[2]);
                ycomp+=tmpxl;
                ycomp.resize_mode(0,dims[0]*2,dims[0]);
                tmpxr.resize_mode(2,dims[2]*2,0);
                tmpxr.resize_mode(0,dims[0]*2,0);
                ycomp+=tmpxr;
                
            }
			Y.set_component(pos,ycomp);
            
		}
		//std::cout<<"time for building y: "<<time (NULL)-begin_time<<" sekunden"<<std::endl;
		return Y;
    }
    
   
    
    
    
    std::vector<Tensor> localProduct_alternative(const std::vector<Tensor>& y,const TTOperator& A){
        
        time_t begin_time = time (NULL);
        
        
        std::vector<Tensor> result;
        
        for (size_t k=0;k<d;k++){
            //built component
            std::vector<Tensor> locY;
            TTTensor Y(xbasis[0].dimensions);
            for (size_t it=0;it<k;it++){
                Y.set_component(it,xbasis[1].get_component(it));
            }
            Y.set_component(k,y[k]);
            for (size_t it=k+1;it<d;it++){
                Y.set_component(it,xbasis[0].get_component(it));
            }
            leftYStack.clear();
            rightYStack.clear();

            leftYStack.emplace_back(Tensor::ones({1,1,1}));
            rightYStack.emplace_back(Tensor::ones({1,1,1}));
        
            for (size_t pos=d-1; pos>0;pos--){
                push_right_stack(pos,Y,A);
            }

            for (size_t corePosition=0;corePosition<d;corePosition++){
                Tensor rhs;
                const Tensor &yi = Y.get_component(corePosition);
                const Tensor &Ai = A.get_component(corePosition);

                Index i1,i2,i3,A1,A2,k1,k2,j2;
                rhs(i1, i2, i3) = leftYStack.back()(i1,A1, k1) * Ai(A1,i2,j2,A2)* yi(k1, j2, k2) * rightYStack.back()(i3,A2, k2);
                
                locY.emplace_back(rhs);
                
                if (corePosition+1 < d) {
                
                    push_left_stack(corePosition,Y,A);
                    rightYStack.pop_back();
                }

            }
            
            
            Index i,j;
           
            
            leftYStack.clear();
            rightYStack.clear();
            if(k==0){
                result=locY;
            }else{
                add(result,locY);
            }
        }
        projection(result);
        //std::cout<<"time for A y product(altenative): "<<time (NULL)-begin_time<<" sekunden"<<std::endl;
  
        return result;
        
        
    }
    
    
    
    std::vector<Tensor> localProduct(const std::vector<Tensor>& y,const TTOperator& A){

        time_t begin_time = time (NULL);
        TTTensor Y=builtTTTensor(y);
        std::vector<Tensor> locY;
        
        
        leftYStack.clear();
		rightYStack.clear();

        leftYStack.emplace_back(Tensor::ones({1,1,1}));
		rightYStack.emplace_back(Tensor::ones({1,1,1}));
       
        for (size_t pos=d-1; pos>0;pos--){
			push_right_stack(pos,Y,A);
		}

		for (size_t corePosition=0;corePosition<d;corePosition++){
            Tensor rhs;
			const Tensor &yi = Y.get_component(corePosition);
            const Tensor &Ai = A.get_component(corePosition);

            Index i1,i2,i3,A1,A2,k1,k2,j2;
            rhs(i1, i2, i3) = leftYStack.back()(i1,A1, k1) * Ai(A1,i2,j2,A2)* yi(k1, j2, k2) * rightYStack.back()(i3,A2, k2);
            
            locY.emplace_back(rhs);
            
            if (corePosition+1 < d) {
            
				push_left_stack(corePosition,Y,A);
				rightYStack.pop_back();
            }

        }
        
        
        Index i,j;
        projection(locY);
        
        leftYStack.clear();
		rightYStack.clear();
        
        //std::cout<<"time for A y product: "<<time (NULL)-begin_time<<" sekunden"<<std::endl;
        return locY;
        
        
    }
    
    std::vector<Tensor> localProduct(const TTOperator& A){
        time_t begin_time = time (NULL);
        TTTensor Y=xbasis[0];
        std::vector<Tensor> locY;
        
        
        leftYStack.clear();
		rightYStack.clear();

        leftYStack.emplace_back(Tensor::ones({1,1,1}));
		rightYStack.emplace_back(Tensor::ones({1,1,1}));
       
        for (size_t pos=d-1; pos>0;pos--){
			push_right_stack(pos,Y,A);
		}

		for (size_t corePosition=0;corePosition<d;corePosition++){
            Tensor rhs;
			const Tensor &yi = Y.get_component(corePosition);
            const Tensor &Ai = A.get_component(corePosition);

            Index i1,i2,i3,A1,A2,k1,k2,j2;
            rhs(i1, i2, i3) = leftYStack.back()(i1,A1, k1) * Ai(A1,i2,j2,A2)* yi(k1, j2, k2) * rightYStack.back()(i3,A2, k2);
            
            locY.emplace_back(rhs);
            
            if (corePosition+1 < d) {
            
				push_left_stack(corePosition,Y,A);
				rightYStack.pop_back();
            }

        }
        
        
        Index i,j;
        projection(locY);
        
        leftYStack.clear();
		rightYStack.clear();
        
        //std::cout<<"time for A x product: "<<time (NULL)-begin_time<<" sekunden"<<std::endl;
        return locY;
        
    }
        
    
    void cg_method(const TTOperator& A,const std::vector<Tensor>& x0,const std::vector<Tensor>& rhs,std::vector<Tensor>& x, size_t maxit, size_t  minit, value_t minerror){//TODO projection!
        
        time_t begin_time = time (NULL);
        auto rhsnorm=frob_norm(rhs);
        //tangentProducts(A);
        value_t error=10e15;
        std::vector<double> errors(minit, 10e17);
        
        
        size_t it=0;
        std::vector<Tensor> r,d,z;
        Index i,j;
        double alpha,beta;
        
        std::vector<Tensor> Ax=localProduct(x,A);
        
        project(Ax,x0);
        //Ax(i&0)=P(i/2,j/2)*Ax(j&0);
        r=rhs;
        add(r,Ax,(-1));
        d=r;
        size_t dims=1;
        for (size_t n=0;n<x.size();n++){
            auto tmp=x[n].dimensions[2]*x[n].dimensions[1]*x[n].dimensions[0];
            dims+=tmp;
        }
        while((it<dims-1)&&(error>minerror*rhsnorm*rhsnorm)&&((it<maxit)&&(errors[errors.size()-minit]>2*error)||(it<minit))){
            
            z=localProduct(d,A);
            project(z,x0);
            
            
            auto tmp1=innerprod(r,r);
            auto tmp2=innerprod(d,z);
            alpha=tmp1/tmp2;
            add(x,d,alpha);
            add(r,z,(-1)*alpha);
            tmp2=innerprod(r,r);
            beta=tmp2/tmp1;
            multiply(d,beta);
            add(d,r);
            error=tmp2;
            errors.push_back(error);
            //std::cout<<"CG     error^2="<<error<<" iteration: "<<it<<" time: "<<time (NULL)-begin_time<<" sekunden"<<std::endl;
            //(rhsnorm*rhsnorm);
            it++;
        }

        std::cout<<"CG     Ax-b: "<<sqrt(error)/frob_norm(rhs)<<"   time for cg method: "<<time (NULL)-begin_time<<" sekunden   "<<it<<"  iterationen"<<std::endl;
        std::cout<<"time for cg method: "<<time (NULL)-begin_time<<" sekunden"<<std::endl;
    }
    
    void krylow(const TTOperator& A,const std::vector<Tensor>& x0,const std::vector<Tensor>& rhs,std::vector<Tensor>& x, size_t maxit, double error){
        time_t begin_time = time (NULL);    
        bool cont=true;
        std::vector<Tensor> res;
        std::vector<Tensor> v;
        auto rhsnorm=frob_norm(rhs);
        double resnorm2=rhsnorm*rhsnorm;
        std::vector<std::vector<Tensor>> V;
        std::vector<std::vector<Tensor>> AV;
        
        if(false){
            res=localProduct(x,A);
            project(res,x0);
            v=res;
            add(res,rhs,(-1));
            resnorm2=innerprod(res,res);
            if(resnorm2>error*rhsnorm*rhsnorm){
                
                
                v=res;
            }
        }
        v=x;
        multiply(x,0);
        size_t it=0;
        
        while((it<maxit)&&(resnorm2>error*rhsnorm*rhsnorm)&&cont){

            auto tmp=localProduct(v,A);
            project(tmp,x0);
           
            
            for(size_t k=0;k<it;k++){
                auto h=innerprod(tmp,AV[k]);                
                add(tmp,AV[k],(-h));
                add(v,V[k],(-h));
            }

            cont=(frob_norm(tmp)>1e-15);
            if (cont){
                auto h=1/frob_norm(tmp);
                multiply(tmp,h);
                multiply(v,h);
                auto bv=innerprod(rhs,tmp);
                add(x,v,bv);
                
                resnorm2-=bv*bv;
                //std::cout<<"krylow error="<<sqrt(resnorm2)/rhsnorm<<" update: "<<sqrt(bv*bv)/rhsnorm<<" iteration: "<<it<<" time: "<<time (NULL)-begin_time<<" sekunden"<<std::endl;  
                V.emplace_back(v);
                AV.emplace_back(tmp);
            v=tmp;
            }
            it++;
        }
//                  auto tmp=localProduct(x,A);
//                  project(tmp,x0);
//                  add(tmp,rhs,(-1));
           //  std::cout<<"krylow Ax-b: "<<frob_norm(tmp)/frob_norm(rhs)<<"  test:  "


            std::cout<<"krylow Ax-b: "<<sqrt(resnorm2)/rhsnorm<<"   time for krylow method: "<<time (NULL)-begin_time<<" sekunden  "<<it<<"  iterationen"<<std::endl;
            
        
    }
    
    TTTensor retraction_ALS(const std::vector<Tensor>& y){
        TTTensor result=xbasis[1];
        TTTensor Y=builtTTTensor(y);
        Tensor tmp=Tensor::ones({1,1});
        for(int pos=d-1;pos>=0;pos--){
            
            result.move_core(pos,true);
            Tensor tmpcore=y[pos];
            if(pos<d-1){
                
                Tensor tmpY=Y.get_component(pos+1);
                Tensor tmpR=result.get_component(pos+1);
                Index i1,i2,k1,k2,k3;
                tmp(i1,i2)=tmpY(i1,k1,k2)*tmpR(i2,k1,k3)*tmp(k2,k3);
                Tensor tmpU=xbasis[1].get_component(pos);
                auto dims=tmpU.dimensions;
                tmpU.resize_mode(2,2*dims[2],dims[2]);
                tmpcore.resize_mode(2,2*dims[2],0);
                tmpcore+=tmpU;
                tmpcore(k1,k2,k3)=tmpcore(k1,k2,i1)*tmp(i1,k3);
                
            }
            
            result.set_component(pos,tmpcore);
        }
        return result;
    }
    
    
    
    
    
    TTTensor retraction(const std::vector<Tensor>& y){
        auto tmp=y;
        tmp[d-1]+=xbasis[1].get_component(d-1); 
        auto tmp1=builtTTTensor(tmp);
        tmp1.round(xbasis[0].ranks());
        return tmp1;        
    }
    
    std::vector<Tensor> vector_transport(const std::vector<Tensor>& y, localProblem help){
        auto tmp=help.builtTTTensor(y);
        return getlocalVector(tmp);
        
    }
    
    
    
    std::vector<Tensor> transportproduct(const std::vector<Tensor>& y,const TTOperator& A,value_t h){
        time_t begin_time = time (NULL);  
        auto tmp0=y;
        multiply(tmp0,h);
        auto tmp=retraction_ALS(tmp0);
        tmp/=frob_norm(tmp);
        localProblem help(tmp);
        auto tmp1= help.localProduct(A);
        auto tmp2=  help.getlocalVector(tmp);
        project(tmp1,tmp2);
        tmp=help.builtTTTensor(tmp1);
        tmp1=getlocalVector(tmp);
        
        std::cout<<"transportproduct: "<<time (NULL)-begin_time<<" sekunden"<<std::endl;
        
        return tmp1;
    }
    
    
    void krylowtransport(const TTOperator& A,const std::vector<Tensor>& x0,const std::vector<Tensor>& rhs,std::vector<Tensor>& x, size_t maxit, double error){
        time_t begin_time = time (NULL);    
        bool cont=true;
        std::vector<Tensor> res;
        std::vector<Tensor> v;
        auto rhsnorm=frob_norm(rhs);
        double resnorm2;
        std::vector<std::vector<Tensor>> V;
        std::vector<std::vector<Tensor>> AV;
        if(true){
            res=transportproduct(x,A,1e-10);
            project(res,x0);
            add(res,rhs,(-1));
            multiply(res,1e10);
            
            v=res;
            add(res,rhs,(-1));
            resnorm2=innerprod(res,res);
            if(resnorm2>error){
                auto h=1/frob_norm(v);
                multiply(v,h);
                multiply(x,h);
                V.emplace_back(x);
                AV.emplace_back(v);
                
                auto bv=innerprod(rhs,v);
                multiply(x,bv);
                    
                
                v=res;
            }
        }
        
        size_t it=1;
        
        while((it<maxit)&&(resnorm2>error*rhsnorm*rhsnorm)&&cont){

            auto tmp=transportproduct(x,A,1e-4);
            project(tmp,x0);
            add(tmp,rhs,(-1));
            multiply(tmp,1e4);
  
            for(size_t k=0;k<it;k++){
                auto h=innerprod(tmp,AV[k]);
                
                add(tmp,AV[k],(-h));
                add(v,V[k],(-h));
            }

            cont=(frob_norm(tmp)>1e-15);
            if (cont){
                auto h=1/frob_norm(tmp);
                multiply(tmp,h);
                multiply(v,h);
                auto bv=innerprod(rhs,tmp);
                std::cout<<bv/rhsnorm<<std::endl;
                add(x,v,bv);
                
                resnorm2-=bv*bv;
                V.emplace_back(v);
                AV.emplace_back(tmp);
            v=tmp;
            }
            it++;
        }
        

//             auto tmp=localProduct(x,A);
//             project(tmp,x0);
//             add(tmp,rhs,(-1));
//             std::cout<<"krylow Ax-b: "<<frob_norm(tmp)/frob_norm(rhs)<<"  test:  "<<sqrt(resnorm2)/frob_norm(rhs)<<"   time for krylow method: "<<time (NULL)-begin_time<<" sekunden  "<<it<<"  iterationen"<<std::endl;
            
        
    }
    
    
};



    
