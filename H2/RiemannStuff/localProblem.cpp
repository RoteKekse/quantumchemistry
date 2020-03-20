#include <xerus.h>


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
    Tensor projectionT;
    Tensor projectionTS;
    
    //projection onto fixed rank  tangentialspace
    void projection(Tensor& y){
        time_t begin_time = time (NULL);
        size_t vectorpos=0;
        for (size_t pos=0;pos<d-1;pos++){
			//TTTensor ypos=xbasis[pos];
			Tensor xi=xbasis[1].get_component(pos);
			auto dims = xi.dimensions;			
			size_t size=dims[0]*dims[1]*dims[2];
			Tensor ycomp=Tensor({size});
			for (size_t k=0;k<size;k++){
				ycomp[k]=y[vectorpos+k];
			}
            ycomp.reinterpret_dimensions(dims);
            Index i1,i2,i3,     j1,j2,j3,       q;
            Tensor tmp;
            tmp(i1,i2,i3)=xi(i1,i2,q)*xi(j1,j2,q)*ycomp(j1,j2,i3);
            ycomp-=tmp;
            ycomp.reinterpret_dimensions({size});
            for (size_t k=0;k<size;k++){
				y[vectorpos+k]=ycomp[k];
			}
			vectorpos+=size;
        }
       // std::cout<<"time for projection: "<<time (NULL)-begin_time<<" sekunden"<<std::endl;
    } 
    
    //projection onto fixed rank and unitcircle tangentialspace
    void projectionS(Tensor& y){
        time_t begin_time = time (NULL);
        size_t vectorpos=0;

        for (size_t pos=0;pos<d;pos++){
			//TTTensor ypos=xbasis[pos];
			Tensor xi=xbasis[1].get_component(pos);
			auto dims = xi.dimensions;			
			size_t size=dims[0]*dims[1]*dims[2];
			Tensor ycomp=Tensor({size});
			for (size_t k=0;k<size;k++){
				ycomp[k]=y[vectorpos+k];
			}
            ycomp.reinterpret_dimensions(dims);
            Index i1,i2,i3,     j1,j2,j3,       q;
            Tensor tmp;
            tmp(i1,i2,i3)=xi(i1,i2,q)*xi(j1,j2,q)*ycomp(j1,j2,i3);
            ycomp-=tmp;
            
            for (size_t k=0;k<size;k++){
				y[vectorpos+k]=ycomp[k];
			}
			vectorpos+=size;
        }
        //std::cout<<"time for projection: "<<time (NULL)-begin_time<<" sekunden"<<std::endl;
        
    }
    
    localProblem(TTTensor& x) 
    : d(x.degree())
    { 
      //  x/=frob_norm(x);
            x.move_core(0,true);
            xbasis.emplace_back(x);
            x.move_core(d-1,true);
            xbasis.emplace_back(x);
        
    }
    
    void update(TTTensor& x){
        xbasis.clear();
    //    x/=frob_norm(x);
        x.move_core(0,true);
        xbasis.emplace_back(x);
        x.move_core(d-1,true);
        xbasis.emplace_back(x);
      
        
    }
    
    
    void built_projection(){
        time_t begin_time = time (NULL);
        
        projectionT=Tensor();
        for(size_t i=0;i<d-1;i++){
            Tensor xi=xbasis[1].get_component(i);
            Tensor Pi;
            
            Index i1,i2,i3,     j1,j2,j3,       q;
            auto dims=xi.dimensions;
            auto I1 =Tensor::identity({dims[2],dims[2]});
            auto I2 =Tensor::identity({dims[0],dims[1],dims[0],dims[1]});
            
            Pi(i1,i2,i3,j1,j2,j3)=I1(i3,j3)*(I2(i1,i2,j1,j2)-(xi(i1,i2,q)*xi(j1,j2,q)));
            
            Pi.reinterpret_dimensions({dims[2]*dims[1]*dims[0],dims[0]*dims[1]*dims[2]});
            
            if(i==0){
                projectionT=Pi;
                
            }else{
                
                projectionT.resize_mode(0,projectionT.dimensions[0]+Pi.dimensions[0],projectionT.dimensions[0]);
            
                Pi.resize_mode(0,projectionT.dimensions[0],0);
            
                projectionT.resize_mode(1,projectionT.dimensions[1]           +Pi.dimensions[1],projectionT.dimensions[1]);
            
                Pi.resize_mode(1,projectionT.dimensions[1],0);
            
                projectionT+=Pi;
            }
        }
        projectionTS=projectionT;
            Tensor xi=xbasis[1].get_component(d-1);
            Tensor Pi;
            
            Index i1,i2,i3,     j1,j2,j3,       q;
            auto dims=xi.dimensions;
            auto I1 =Tensor::identity({dims[2],dims[2]});
            auto I2 =Tensor::identity({dims[0],dims[1],dims[0],dims[1]});
            auto I3 =Tensor::identity({dims[0]*dims[1]*dims[2],dims[0]*dims[1]*dims[2]});
            
            
            Pi(i1,i2,i3,j1,j2,j3)=I1(i3,j3)*(I2(i1,i2,j1,j2)-(xi(i1,i2,q)*xi(j1,j2,q)));
            
            Pi.reinterpret_dimensions({dims[2]*dims[1]*dims[0],dims[0]*dims[1]*dims[2]});
            
            projectionT.resize_mode(0,projectionT.dimensions[0]+Pi.dimensions[0],projectionT.dimensions[0]);
            
            Pi.resize_mode(0,projectionT.dimensions[0],0);
            projectionTS.resize_mode(0,projectionT.dimensions[0],projectionTS.dimensions[0]);
            I3.resize_mode(0,projectionT.dimensions[0],0);
            projectionT.resize_mode(1,projectionT.dimensions[1]+Pi.dimensions[1],projectionT.dimensions[1]);
            
            Pi.resize_mode(1,projectionT.dimensions[1],0);
            projectionTS.resize_mode(1,projectionT.dimensions[1],projectionTS.dimensions[1]);
            I3.resize_mode(1,projectionT.dimensions[1],0);
            projectionT+=I3;
            projectionTS+=Pi;
        
            //std::cout<<"time for projection matrix: "<<time (NULL)-begin_time<<" sekunden"<<std::endl;
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
	
	Tensor getBlockDiagOp(const TTOperator& A){
        time_t begin_time = time (NULL);
        //first build every needed stack
        leftAStack.emplace_back(Tensor::ones({1,1,1}));
		rightAStack.emplace_back(Tensor::ones({1,1,1}));
        
        for (size_t pos=0; pos<d-1;pos++){
			push_left_stack(pos,A);
        }
		for (size_t pos=d-1; pos>0;pos--){
			push_right_stack(pos,A);
		}
		
		//now build matrix
		
		Tensor Aloc=Tensor();
        for(size_t i=0;i<d;i++){
            Tensor Acomp;
            Index i1,i2,i3,j1,j2,j3,k1,k2;
            Tensor left=leftAStack[i];
            Tensor right=rightAStack[d-i-1];
            Tensor Ai= A.get_component(i);
                    
            Acomp(i1,i2,i3,j1,j2,j3)=left(i1,k1,j1)*Ai(k1,i2,j2,k2)*right(i3,k2,j3);
            auto d1=Acomp.dimensions[0]*Acomp.dimensions[1]*Acomp.dimensions[2];
            auto d2=Acomp.dimensions[3]*Acomp.dimensions[4]*Acomp.dimensions[5];
                    
            Acomp.reinterpret_dimensions({d1,d2});   
            if(i==0){
                Aloc=Acomp;
            }else{
                auto dims =Acomp.dimensions;
                auto dims2=Aloc.dimensions;
                    
                Aloc.resize_mode(1,dims[1]+dims2[1],dims2[1]);
                Acomp.resize_mode(1,dims[1]+dims2[1],0);
                Aloc.resize_mode(0,dims[0]+dims2[0],dims2[0]);
                Acomp.resize_mode(0,dims[0]+dims2[0],0);
                Aloc+=Acomp;
            }
            
        }
        leftAStack.clear();
		rightAStack.clear();
		return Aloc;		
    }
	
    Tensor getlocalOperator(const TTOperator& A){
        
        time_t begin_time = time (NULL);
        
        
        

        
        leftAStack.emplace_back(Tensor::ones({1,1,1}));
		rightAStack.emplace_back(Tensor::ones({1,1,1}));
        
        
            
        for (size_t pos=0; pos<d-1;pos++){
			push_left_stack(pos,A);
        }
		for (size_t pos=d-1; pos>0;pos--){
			push_right_stack(pos,A);
		}
		//if possible  do in parallel
		for (size_t n=2;n<d;n++){
            Tensor tmpA,Ai,xL,xR;
            Ai=A.get_component(n-1);
            xL=xbasis[1].get_component(n-1);
            xR=xbasis[0].get_component(n-1);
            
            Index i1,i2,i3,     j1,j2,j3,   k1,k2;
            
            tmpA(i1,i2,i3, j1,j2,j3)=   xR(i1, k1, j1)*Ai(i2, k1, k2, j2)*xL(i3, k2, j3);
            
            middleAStack.emplace_back(tmpA);
            
     
        }
        
        Tensor Aloc;
        for(size_t i=0;i<d;i++){
            
            Tensor Aloc1;
            Tensor Acomp;
            const TTTensor &xi =xbasis[0];
			const TTTensor &xj =xbasis[1];		
    
			Tensor xicomp, xjcomp,Ai;

            Tensor Mi,Mj;
            
            
            

            //diagonal component
            Index i1,i2,i3,j1,j2,j3,l1,l2,l3,r1,r2,r3,k1,k2,k3,k4, n1,n2,n3;
            Tensor left=leftAStack[i];
            Tensor right=rightAStack[d-i-1];
            Ai= A.get_component(i);
                    
            Aloc1(i1,i2,i3,j1,j2,j3)=0.5*left(i1,k1,j1)*Ai(k1,i2,j2,k2)*right(i3,k2,j3);
            
            size_t d1,d2;
            d1=Aloc1.dimensions[0]*Aloc1.dimensions[1]*Aloc1.dimensions[2];
            d2=Aloc1.dimensions[3]*Aloc1.dimensions[4]*Aloc1.dimensions[5];
                    
            Aloc1.reinterpret_dimensions({d1,d2}); 
            
            
            
            
            
            if(i<d-1){
                size_t j =i+1;
                xicomp= xi.get_component(j);
                xjcomp= xj.get_component(i);	
                Tensor Aj = A.get_component(j);
                Ai= A.get_component(i);
                using xerus::misc::operator<<;
                Mi(i2,k1,r1,k2,r2)=Ai(k1,i2,n2,k2)*xjcomp(r1,n2,r2);
                Mj(j2,l1,k1,l2,k2)=xicomp(l1,n1,l2)*Aj(k1,n1,j2,k2); 
                
                
                    
                Tensor left=leftAStack[i];
                Tensor right=rightAStack[d-j-1];
                //Tensor middle=middleAStack[i][j];
                    
                Tensor xi=xbasis[0].get_component(i);
                Tensor xj=xbasis[1].get_component(j);
                    
                Acomp(i1,i2,i3,j1,j2,j3)=left(i1,k1,r1)*Mi(i2,k1,r1,k2,j1)*Mj(j2,i3,k2,l2,k4)*right(l2,k4,j3);
                
                size_t d1,d2;
                d1=Acomp.dimensions[0]*Acomp.dimensions[1]*Acomp.dimensions[2];
                d2=Acomp.dimensions[3]*Acomp.dimensions[4]*Acomp.dimensions[5];
                
                Acomp.reinterpret_dimensions({d1,d2}); 
                
                auto dims =Acomp.dimensions;
                auto dims2=Aloc1.dimensions;
                    
                Aloc1.resize_mode(1,dims[1]+dims2[1],dims2[1]);
                Acomp.resize_mode(1,dims[1]+dims2[1],0);
                    //std::cout<<Aloc1.to_string()<<std::endl;
                    //std::cout<<Acomp.to_string()<<std::endl;
          
                Aloc1+=Acomp;
                
            }
            Tensor tmp,tmp1,tmp2;
            for (size_t j= i+2; j<d;j++){
                
                if(j==i+2){
                    tmp=middleAStack[i];
                }else{
                  
                    Index i1,i2,i3,     j1,j2,j3,   k1,k2,k3;
                    tmp1=tmp;
                    tmp2=middleAStack[j-2];
                    
                    
                    tmp(i1,i2,i3, j1,j2,j3)=tmp1(i1,i2,i3, k1,k2,k3)*tmp2(k1,k2,k3, j1,j2,j3);
                    
                }
              
                    
                Tensor left=leftAStack[i];
                Tensor right=rightAStack[d-j-1];
                    
                
                
                 xicomp= xi.get_component(j);
                xjcomp= xj.get_component(i);	
                
                
                Tensor xi=xbasis[0].get_component(i);
                Tensor xj=xbasis[1].get_component(j);
                    
               
                Tensor Aj = A.get_component(j);
                Ai= A.get_component(i);
                
                Mi(i2,k1,r1,k2,r2)=Ai(k1,i2,n2,k2)*xjcomp(r1,n2,r2);
                Mj(j2,l1,k1,l2,k2)=xicomp(l1,n1,l2)*Aj(k1,n1,j2,k2);
                
                
                Acomp(i1,i2,i3,j1,j2,j3)=left(i1,k1,r1)*Mi(i2,k1,r1,k2,r2)*tmp(i3,k2,r2,l1,k3,j1)*Mj(j2,l1,k3,l2,k4)*right(l2,k4,j3);
                
                
                size_t d1,d2;
                d1=Acomp.dimensions[0]*Acomp.dimensions[1]*Acomp.dimensions[2];
                d2=Acomp.dimensions[3]*Acomp.dimensions[4]*Acomp.dimensions[5];
                
                Acomp.reinterpret_dimensions({d1,d2}); 
                
                auto dims =Acomp.dimensions;
                auto dims2=Aloc1.dimensions;
                    
                Aloc1.resize_mode(1,dims[1]+dims2[1],dims2[1]);
                Acomp.resize_mode(1,dims[1]+dims2[1],0);
                    //std::cout<<Aloc1.to_string()<<std::endl;
                    //std::cout<<Acomp.to_string()<<std::endl;
          
                Aloc1+=Acomp;
                
            }
            
            if(i==0){
                Aloc=Aloc1;
            }else{
                auto dims =Aloc.dimensions;
				auto dims2=Aloc1.dimensions;
            
				Aloc.resize_mode(0,dims[0]+dims2[0],dims[0]);
				Aloc1.resize_mode(0,dims[0]+dims2[0],0);
                Aloc1.resize_mode(1,dims[1],0);
				Aloc+=Aloc1;
                
            }
            
            
        }
        Index i,j;
        

        Aloc(i,j)=Aloc(i,j)+Aloc(j,i);
        
        
        std::cout<<"time for A components: "<<time (NULL)-begin_time<<" sekunden"<<std::endl;
        middleAStack.clear();
        leftAStack.clear();
		rightAStack.clear();
        return Aloc;
        
        
        
    }

    Tensor getlocalVector(const TTTensor& y){
        time_t begin_time = time (NULL);
        Tensor locY=Tensor();
        
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
            
            rhs.reinterpret_dimensions({rhs.dimensions[2]*rhs.dimensions[1]*rhs.dimensions[0]});
            
            if(corePosition==0){
                locY=rhs;
            }else{
            
                locY.resize_mode(0,locY.dimensions[0]+rhs.dimensions[0],locY.dimensions[0]);
            
                rhs.resize_mode(0,locY.dimensions[0],0);
            
                locY+=rhs;
            }
            if (corePosition+1 < d) {
            
				push_left_stack(corePosition,y);
				rightYStack.pop_back();
            }
        }
       // std::cout<<"time for y components: "<<time (NULL)-begin_time<<" sekunden"<<std::endl;
        return locY;
        
    }
    
    TTTensor builtTTTensor( Tensor& y){
        time_t begin_time = time (NULL);
        
        TTTensor Y(xbasis[0].dimensions);
        
        
        size_t vectorpos=0;
		for (size_t pos=0;pos<d;pos++){
			//TTTensor ypos=xbasis[pos];
			
			auto dims = xbasis[0].get_component(pos).dimensions;			
			size_t size=dims[0]*dims[1]*dims[2];
			Tensor ycomp=Tensor({size});
			for (size_t k=0;k<size;k++){
				ycomp[k]=y[vectorpos+k];
			}
            ycomp.reinterpret_dimensions(dims);
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
            Index i,j;
			//ypos.set_component(pos,ycomp);
			Y.set_component(pos,ycomp);
			vectorpos+=size;
            
		}
		//std::cout<<"time for building y: "<<time (NULL)-begin_time<<" sekunden"<<std::endl;
		return Y;
    }
    
    
    Tensor tangentAProduct( Tensor& y,const TTOperator& A){
        time_t begin_time = time (NULL);
        
        TTTensor Y(xbasis[0].dimensions);
        
        
        size_t vectorpos=0;
		for (size_t pos=0;pos<d;pos++){
			//TTTensor ypos=xbasis[pos];
			
			auto dims = xbasis[0].get_component(pos).dimensions;			
			size_t size=dims[0]*dims[1]*dims[2];
			Tensor ycomp=Tensor({size});
			for (size_t k=0;k<size;k++){
				ycomp[k]=y[vectorpos+k];
			}
			Tensor tmpxl,tmpxr;
            ycomp.reinterpret_dimensions(dims);
            if(pos<d-1){
                tmpxl=Axleft[pos];
            }
            if(pos>0){
                tmpxr=Axright[pos-1];
            }
            
            Tensor tmpA=A.get_component(pos);
            
            Index a1,a2,x1,x2,i,j;
            ycomp(a1,x1,i,a2,x2)=tmpA(a1,i,j,a2)*ycomp(x1,j,x2);
            
            dims=ycomp.dimensions;
            
            ycomp.reinterpret_dimensions({dims[0]*dims[1],dims[2],dims[3]*dims[4]});
            
            dims=ycomp.dimensions;
            
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
            
			//ypos.set_component(pos,ycomp);
			Y.set_component(pos,ycomp);
			vectorpos+=size;
            
		}
		
		Tensor result =getlocalVector(Y);
        Index i,j;
        projection(result);
		//std::cout<<"time for building y: "<<time (NULL)-begin_time<<" sekunden"<<std::endl;
		return result;
    }
    
    
    Tensor localProduct(Tensor& y,const TTOperator& A){

        time_t begin_time = time (NULL);
        TTTensor Y=builtTTTensor(y);
        Tensor locY=Tensor();
        
        
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
            
            rhs.reinterpret_dimensions({rhs.dimensions[2]*rhs.dimensions[1]*rhs.dimensions[0]});
      
            if(corePosition==0){
                locY=rhs;
            }else{
            
                locY.resize_mode(0,locY.dimensions[0]+rhs.dimensions[0],locY.dimensions[0]);
            
                rhs.resize_mode(0,locY.dimensions[0],0);
            
                locY+=rhs;
            }
            if (corePosition+1 < d) {
            
				push_left_stack(corePosition,Y,A);
				rightYStack.pop_back();
            }

        }
        
        
        Index i,j;
        projection(locY);
        
        leftYStack.clear();
		rightYStack.clear();
        
  //      std::cout<<"time for A y product: "<<time (NULL)-begin_time<<" sekunden"<<std::endl;
        return locY;
        
        
    }
    
    Tensor localProduct(const TTOperator& A){
        time_t begin_time = time (NULL);
        
        Index i,j;
        Tensor result;
        TTTensor tmp;
        
        tmp(i&0)=A(i/2,j/2)*xbasis[0](j&0);
        
        result=getlocalVector(tmp);
        projection(result);
        
  //      std::cout<<"time for product: "<<time (NULL)-begin_time<<" sekunden"<<std::endl;
        
        return result;
        
        
    }
    
    void tangentProducts(const TTOperator& A){
        
        time_t begin_time = time (NULL);
        
        Axleft.clear();
        Axright.clear();
    
        for(size_t pos=0;pos<d-1;pos++){
            Tensor tmp,tmpA,tmpx;
            tmpA=A.get_component(pos);
            tmpx=xbasis[1].get_component(pos);
            
            Index a1,a2,x1,x2,i,j;
            tmp(a1,x1,i,a2,x2)=tmpA(a1,i,j,a2)*tmpx(x1,j,x2);
            
            auto dims=tmp.dimensions;
            
            tmp.reinterpret_dimensions({dims[0]*dims[1],dims[2],dims[3]*dims[4]});
            Axleft.emplace_back(tmp);
            
        }
        for(size_t pos=1;pos<d;pos++){
            Tensor tmp,tmpA,tmpx;
            tmpA=A.get_component(pos);
            tmpx=xbasis[0].get_component(pos);
            
            Index a1,a2,x1,x2,i,j;
            tmp(a1,x1,i,a2,x2)=tmpA(a1,i,j,a2)*tmpx(x1,j,x2);
           
            auto dims=tmp.dimensions; 
            
            tmp.reinterpret_dimensions({dims[0]*dims[1],dims[2],dims[3]*dims[4]});
            
            Axright.emplace_back(tmp);
        }
        
        std::cout<<"time for building Ax comps: "<<time (NULL)-begin_time<<" sekunden"<<std::endl;
        
    }
    
    
    void cg_method3(const TTOperator& A, Tensor& rhs, Tensor& x, size_t maxit, size_t  minit, value_t minerror){//TODO projection!
        
        time_t begin_time = time (NULL);
        
        //tangentProducts(A);
        value_t error=100;
        size_t it=0;
        Tensor r,d,z;
        Index i,j;
        double alpha,beta,rhsnorm;
        rhsnorm=frob_norm(rhs);
        Tensor Ax=localProduct(x,A);
        Tensor tmp;
       // tmp(i)=x0(i)*x0(j)*Ax(j);
       // Ax-=tmp;
        //Ax(i&0)=P(i/2,j/2)*Ax(j&0);
        r(i&0)=rhs(i&0)-Ax(i&0);
        d=r;
        size_t dims=1;
        for (size_t n=0;n<x.dimensions.size();n++){
            dims*=x.dimensions[n];
        }
        while((it<dims-1)&&(((error>minerror*minerror)&&(it<maxit))||(it<minit))){
            z=localProduct(d,A);
            Tensor tmp;
     //       tmp(i)=x0(i)*x0(j)*z(j);
     //       z-=tmp;
            //z(i&0)=P(i/2,j/2)*z(j&0);
            Tensor tmp1,tmp2;
            tmp1()=r(i&0)*r(i&0);
            tmp2()=d(i&0)*z(i&0);
            alpha=tmp1[0]/tmp2[0];
            x+=alpha*d;
            projection(x);
            r-=alpha*z;
            tmp2()=r(i&0)*r(i&0);
            beta=tmp2[0]/tmp1[0];
            d=r+beta*d;
        
            projection(d);
            error=tmp2[0];//(rhsnorm*rhsnorm);
            it++;
        }
        std::cout<<"error^2="<<error<<" iterations: "<<it<<std::endl;
        std::cout<<"time for cg method: "<<time (NULL)-begin_time<<" sekunden"<<std::endl;
    }    
    
    void cg_method(const TTOperator& A,Tensor& x0, Tensor& rhs, Tensor& x, size_t maxit, size_t  minit, value_t minerror){//TODO projection!
        
        time_t begin_time = time (NULL);
        
        //tangentProducts(A);
        value_t error=100;
        size_t it=0;
        Tensor r,d,z;
        Index i,j;
        double alpha,beta,rhsnorm;
        rhsnorm=frob_norm(rhs);
        Tensor Ax=localProduct(x,A);
        Tensor tmp;
        tmp(i)=x0(i)*x0(j)*Ax(j);
        Ax-=tmp;
        //Ax(i&0)=P(i/2,j/2)*Ax(j&0);
        r(i&0)=rhs(i&0)-Ax(i&0);
        d=r;
        size_t dims=1;
        for (size_t n=0;n<x.dimensions.size();n++){
            dims*=x.dimensions[n];
        }
        while((it<dims-1)&&(((error>minerror*minerror)&&(it<maxit))||(it<minit))){
            z=localProduct(d,A);
            Tensor tmp;
            tmp(i)=x0(i)*x0(j)*z(j);
            z-=tmp;
            //z(i&0)=P(i/2,j/2)*z(j&0);
            Tensor tmp1,tmp2;
            tmp1()=r(i&0)*r(i&0);
            tmp2()=d(i&0)*z(i&0);
            alpha=tmp1[0]/tmp2[0];
            x+=alpha*d;
           // projection(x);
            r-=alpha*z;
            tmp2()=r(i&0)*r(i&0);
            beta=tmp2[0]/tmp1[0];
            d=r+beta*d;
        
           // projection(d);

            Tensor tmp3;
            tmp3() = d(i&0) * z(i&0);


            error=tmp2[0];//(rhsnorm*rhsnorm);
            std::cout<<"error^2="<<error<<" iterations: "<<it<< " orth? " << tmp3[0] << " dz = " << d.frob_norm() * z.frob_norm()<<std::endl;

            it++;
        }
        std::cout<<"error^2="<<error<<" iterations: "<<it<<std::endl;
        std::cout<<"time for cg method: "<<time (NULL)-begin_time<<" sekunden"<<std::endl;
    }
    
    void cg_method2(const TTOperator& A,Tensor& P, Tensor& rhs, Tensor& x, size_t maxit, size_t  minit, value_t minerror){
        
        time_t begin_time = time (NULL);
        
        tangentProducts(A);
        value_t error=100;
        size_t it=0;
        Tensor r,d,z;
        Index i,j;
        double alpha,beta,rhsnorm;
        rhsnorm=frob_norm(rhs);
        Tensor Ax=tangentAProduct(x,A);
        Ax(i&0)=P(i/2,j/2)*Ax(j&0);
        r(i&0)=rhs(i&0)-Ax(i&0);
        d=r;
        size_t dims=1;
        for (size_t n=0;n<x.dimensions.size();n++){
            dims*=x.dimensions[n];
        }
        while((it<dims-1)&&(((error>minerror*minerror)&&(it<maxit))||(it<minit))){
            z=tangentAProduct(d,A);
            z(i&0)=P(i/2,j/2)*z(j&0);
            Tensor tmp1,tmp2;
            tmp1()=r(i&0)*r(i&0);
            tmp2()=d(i&0)*z(i&0);
            alpha=tmp1[0]/tmp2[0];
            x+=alpha*d;
            projection(x);
            r-=alpha*z;
            tmp2()=r(i&0)*r(i&0);
            beta=tmp2[0]/tmp1[0];
            d=r+beta*d;
        
            projection(d);
            error=tmp2[0];//(rhsnorm*rhsnorm);
            it++;
        }
    std::cout<<"error^2="<<error<<" iterations: "<<it<<std::endl;
    std::cout<<"time for cg method: "<<time (NULL)-begin_time<<" sekunden"<<std::endl;
    }
    
    
    
    
    
    
};
