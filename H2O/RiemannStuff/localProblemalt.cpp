#include <xerus.h>


using namespace xerus;



class localProblem{
    const size_t d;
    std::vector<TTTensor> xbasis;
    
    std::vector<Tensor> leftAStack;
    std::vector<Tensor> rightAStack;
    std::vector<std::vector<Tensor>> middleAStack;
    
    std::vector<Tensor> leftYStack;
    std::vector<Tensor> rightYStack;
    
public:
    Tensor projection; //projection onto fixed rank  tangentialspace
    Tensor projectionS;//projection onto fixed rank and unitcircle tangentialspace
    
    localProblem(TTTensor& x) 
    : d(x.degree())
    { 
        x/=frob_norm(x);
        for (size_t pos=0; pos<d;pos++){
            x.move_core(pos,true);
            xbasis.emplace_back(x);
        }
        built_projection();
    }
    
    void update(TTTensor& x){
        xbasis.clear();
        x/=frob_norm(x);
        for (size_t pos=0; pos<d;pos++){
            x.move_core(pos,true);
            xbasis.emplace_back(x);
        }
        built_projection();
        
    }
    
    
    void built_projection(){
        time_t begin_time = time (NULL);
        
        projection=Tensor();
        for(size_t i=0;i<d-1;i++){
            Tensor xi=xbasis[d-1].get_component(i);
            Tensor Pi;
            
            Index i1,i2,i3,     j1,j2,j3,       q;
            auto dims=xi.dimensions;
            auto I1 =Tensor::identity({dims[2],dims[2]});
            auto I2 =Tensor::identity({dims[0],dims[1],dims[0],dims[1]});
            
            Pi(i1,i2,i3,j1,j2,j3)=I1(i3,j3)*(I2(i1,i2,j1,j2)-(xi(i1,i2,q)*xi(j1,j2,q)));
            
            Pi.reinterpret_dimensions({dims[2]*dims[1]*dims[0],dims[0]*dims[1]*dims[2]});
            
            if(i==0){
                projection=Pi;
                
            }else{
                
                projection.resize_mode(0,projection.dimensions[0]+Pi.dimensions[0],projection.dimensions[0]);
            
                Pi.resize_mode(0,projection.dimensions[0],0);
            
                projection.resize_mode(1,projection.dimensions[1]           +Pi.dimensions[1],projection.dimensions[1]);
            
                Pi.resize_mode(1,projection.dimensions[1],0);
            
                projection+=Pi;
            }
        }
        projectionS=projection;
            Tensor xi=xbasis[d-1].get_component(d-1);
            Tensor Pi;
            
            Index i1,i2,i3,     j1,j2,j3,       q;
            auto dims=xi.dimensions;
            auto I1 =Tensor::identity({dims[2],dims[2]});
            auto I2 =Tensor::identity({dims[0],dims[1],dims[0],dims[1]});
            auto I3 =Tensor::identity({dims[0]*dims[1]*dims[2],dims[0]*dims[1]*dims[2]});
            
            
            Pi(i1,i2,i3,j1,j2,j3)=I1(i3,j3)*(I2(i1,i2,j1,j2)-(xi(i1,i2,q)*xi(j1,j2,q)));
            
            Pi.reinterpret_dimensions({dims[2]*dims[1]*dims[0],dims[0]*dims[1]*dims[2]});
            
            projection.resize_mode(0,projection.dimensions[0]+Pi.dimensions[0],projection.dimensions[0]);
            
            Pi.resize_mode(0,projection.dimensions[0],0);
            projectionS.resize_mode(0,projection.dimensions[0],projectionS.dimensions[0]);
            I3.resize_mode(0,projection.dimensions[0],0);
            projection.resize_mode(1,projection.dimensions[1]+Pi.dimensions[1],projection.dimensions[1]);
            
            Pi.resize_mode(1,projection.dimensions[1],0);
            projectionS.resize_mode(1,projection.dimensions[1],projectionS.dimensions[1]);
            I3.resize_mode(1,projection.dimensions[1],0);
            projection+=I3;
            projectionS+=Pi;
        
            //std::cout<<"time for projection matrix: "<<time (NULL)-begin_time<<" sekunden"<<std::endl;
    }
    
    void push_left_stack(const size_t _position,TTTensor& y) {
		Index i1, i2, i3, j1 , j2, j3, k1, k2;
		const Tensor &xi = xbasis[d-1].get_component(_position);
		const Tensor &yi = y.get_component(_position);

		
		Tensor tmpy;
		tmpy(i1, i2) = leftYStack.back()(j1, j2)
				*xi(j1, k1, i1)*yi(j2, k1, i2);
		leftYStack.emplace_back(std::move(tmpy));

	}
	
	
	void push_right_stack(const size_t _position,TTTensor& y){
        Index i1, i2, i3, j1 , j2, j3, k1, k2;
        const Tensor &xi = xbasis[0].get_component(_position);
		const Tensor &yi = y.get_component(_position);
		
		Tensor tmpy;
		tmpy(i1, i2) = xi(i1, k1, j1)*yi(i2, k1, j2)
				*rightYStack.back()(j1, j2);
		rightYStack.emplace_back(std::move(tmpy));

	}
    
    void push_left_stack(const size_t _position,const TTOperator& A) {
		Index i1, i2, i3, j1 , j2, j3, k1, k2;
		const Tensor &xi = xbasis[d-1].get_component(_position);
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
        //first build every stack
        
        
        middleAStack.resize(d,std::vector<Tensor>(d,Tensor()));
        
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
            xL=xbasis[d-1].get_component(n-1);
            xR=xbasis[0].get_component(n-1);
            
            Index i1,i2,i3,     j1,j2,j3,   k1,k2;
            
            tmpA(i1,i2,i3, j1,j2,j3)=   xR(i1, k1, j1)*Ai(i2, k1, k2, j2)*xL(i3, k2, j3);
            
            middleAStack[n-2][n]=tmpA;
            
        }
        
        
            
        //if possible  do in parallel
        for (size_t i=0;i<d-3;i++){
            for (size_t j=i+3;j<d;j++){
                Index i1,i2,i3,     j1,j2,j3,   k1,k2,k3;
                Tensor tmp,tmp1,tmp2;
                tmp1=middleAStack[i][j-1];
                
                tmp2=middleAStack[j-2][j];
                
                tmp(i1,i2,i3, j1,j2,j3)=tmp1(i1,i2,i3, k1,k2,k3)*tmp2(k1,k2,k3, j1,j2,j3);
                middleAStack[i][j]=tmp;
                
            }
        }
		
        
         
		//if possible  do in parallel
		for (size_t n=2;n<d;n++){
            Tensor tmpA,Ai,xL,xR;
            Ai=A.get_component(n-1);
            xL=xbasis[d-1].get_component(n-1);
            xR=xbasis[0].get_component(n-1);
            
            Index i1,i2,i3,     j1,j2,j3,   k1,k2;
            
            tmpA(i1,i2,i3, j1,j2,j3)=   xL(i1, k1, j1)*Ai(i2, k1, k2, j2)*xR(i3, k2, j3);
            
            middleAStack[n][n-2]=tmpA;
            
        }
        
        
            
        //if possible  do in parallel
        for (size_t i=0;i<d-3;i++){
            for (size_t j=i+3;j<d;j++){
                Index i1,i2,i3,     j1,j2,j3,   k1,k2,k3;
                middleAStack[j][i](i1,i2,i3, j1,j2,j3)=middleAStack[j-1][i](i1,i2,i3, k1,k2,k3)*middleAStack[j][j-2](k1,k2,k3, j1,j2,j3);
                
            }
        }
        //building stacks finished
        std::cout<<"time for A stacks: "<<time (NULL)-begin_time<<" sekunden"<<std::endl;
        
        
        //building matrix now
        
        //could be done in parallel too
        Tensor Aloc=Tensor();
        for(size_t i=0;i<d;i++){
            Tensor Aloc1=Tensor();
            for(size_t j=0;j<d;j++){
                Tensor Acomp;
                
                
                
            const TTTensor &xi =xbasis[i];
			const TTTensor &xj =xbasis[j];		
			Index l1,l2,k1,k2,k3,k4,r1,r2,  n1,n2,n3,  i1,i2,i3, j1,j2,j3;	
			Tensor xicomp, xjcomp,Ai;
			Tensor middle;
			//do we need a middle stack?
            Tensor Mi,Mj;
            if(j!=i){
            
                if((i-j==1) or (j-i==1)){
                    size_t d1,d2;
                    Tensor I1,I2;				
                    xicomp=xi.get_component(std::min(i,j));
                    Ai= A.get_component(std::min(i,j));
                    d1=xicomp.dimensions[2];
                    d2=Ai.dimensions[3];
                    I1=Tensor::identity({d1,d1});
                    I2=Tensor::identity({d2,d2});
                    middle(l1,k1,r1,l2,k2,r2)=I1(l1,l2)*I2(k1,k2)*I1(r1,r2);
                }
            
			
			
                xicomp= xi.get_component(j);
                xjcomp= xj.get_component(i);	
                Tensor Aj = A.get_component(j);
                Ai= A.get_component(i);
                using xerus::misc::operator<<;
                Mi(i2,k1,r1,k2,r2)=Ai(k1,i2,n2,k2)*xjcomp(r1,n2,r2);
                Mj(j2,l1,k1,l2,k2)=xicomp(l1,n1,l2)*Aj(k1,n1,j2,k2);
            }    
                
                
                if(i==j){
                    Index i1,i2,i3,j1,j2,j3,k1,k2;
                    Tensor left=leftAStack[i];
                    Tensor right=rightAStack[d-j-1];
                    Tensor Ai= A.get_component(i);
                    
                    Acomp(i1,i2,i3,j1,j2,j3)=left(i1,k1,j1)*Ai(k1,i2,j2,k2)*right(i3,k2,j3);
                    
                    
        
            
        
                    
                }else if(i==j+1){
                    Index i1,i2,i3,j1,j2,j3,l1,l2,l3,r1,r2,r3,k1;
                    
                    Tensor left=leftAStack[j];
                    Tensor right=rightAStack[d-i-1];
                    
                    
                    Tensor Ai= A.get_component(i);
                    Tensor Aj= A.get_component(j);
                    
                    Tensor xi=xbasis[d-1].get_component(i);
                    Tensor xj=xbasis[0].get_component(j);
                    
                    
                    Acomp(i1,i2,i3,j1,j2,j3)=left(l1,k1,j1)*Mj(j2,l1,k1,l2,k2)*middle(l2,k2,j3,i1,k3,r1)*Mi(i2,k3,r1,k4,r2)*right(i3,k4,r2);
                 
                    
                    
                    
                }else if(j==i+1){
                    Index i1,i2,i3,j1,j2,j3,l1,l2,l3,r1,r2,r3,k1;
                    
                    Tensor left=leftAStack[i];
                    Tensor right=rightAStack[d-j-1];
                    //Tensor middle=middleAStack[i][j];
                    
                    Tensor Ai= A.get_component(i);
                    Tensor Aj= A.get_component(j);
                    
                    Tensor xi=xbasis[0].get_component(i);
                    Tensor xj=xbasis[d-1].get_component(j);
                    
                    Acomp(i1,i2,i3,j1,j2,j3)=left(i1,k1,r1)*Mi(i2,k1,r1,k2,r2)*middle(i3,k2,r2,l1,k3,j1)*Mj(j2,l1,k3,l2,k4)*right(l2,k4,j3);
                    
                    
                    
                }else if (i>j){
                    Index i1,i2,i3,j1,j2,j3,l1,l2,l3,r1,r2,r3,k1,k2,k3,k4;
                    
                    Tensor left=leftAStack[j];
                    Tensor right=rightAStack[d-i-1];
                    Tensor middle=middleAStack[i][j];
                    
                    Tensor Ai= A.get_component(i);
                    Tensor Aj= A.get_component(j);
                    
                    Tensor xi=xbasis[d-1].get_component(i);
                    Tensor xj=xbasis[0].get_component(j);
                
                    Acomp(i1,i2,i3,j1,j2,j3)=left(l1,k1,j1)*Mj(j2,l1,k1,l2,k2)*middle(l2,k2,j3,i1,k3,r1)*Mi(i2,k3,r1,k4,r2)*right(i3,k4,r2);
                    
                    
                }else if(j>i){
                    Index i1,i2,i3,j1,j2,j3,l1,l2,l3,r1,r2,r3,k1,k2,k3,k4;
                    
                    Tensor left=leftAStack[i];
                    Tensor right=rightAStack[d-j-1];
                    Tensor middle=middleAStack[i][j];
                    
                    Tensor Ai= A.get_component(i);
                    Tensor Aj= A.get_component(j);
                    
                    Tensor xi=xbasis[0].get_component(i);
                    Tensor xj=xbasis[d-1].get_component(j);
                    
                    
                    Acomp(i1,i2,i3,j1,j2,j3)=left(i1,k1,r1)*Mi(i2,k1,r1,k2,r2)*middle(i3,k2,r2,l1,k3,j1)*Mj(j2,l1,k3,l2,k4)*right(l2,k4,j3);
                    
                }
                size_t d1,d2;
                d1=Acomp.dimensions[0]*Acomp.dimensions[1]*Acomp.dimensions[2];
                d2=Acomp.dimensions[3]*Acomp.dimensions[4]*Acomp.dimensions[5];
                    
                Acomp.reinterpret_dimensions({d1,d2});    
                    
                    
                    
                
                
                //insert into row
                
                if(j==0){
                    Aloc1=Acomp;
                }else{
                    auto dims =Acomp.dimensions;
                    auto dims2=Aloc1.dimensions;
                    
                    Aloc1.resize_mode(1,dims[1]+dims2[1],dims2[1]);
                    Acomp.resize_mode(1,dims[1]+dims2[1],0);
                    //std::cout<<Aloc1.to_string()<<std::endl;
                    //std::cout<<Acomp.to_string()<<std::endl;
          
                    Aloc1+=Acomp;
                }
            
            
            }
            
            //insert into matrix
            if(i==0){
                Aloc=Aloc1;
            }else{
                auto dims =Aloc.dimensions;
				auto dims2=Aloc1.dimensions;
            
				Aloc.resize_mode(0,dims[0]+dims2[0],dims[0]);
				Aloc1.resize_mode(0,dims[0]+dims2[0],0);
				Aloc+=Aloc1;
            }
        }
        std::cout<<"time for A components: "<<time (NULL)-begin_time<<" sekunden"<<std::endl;
        middleAStack.clear();
        leftAStack.clear();
		rightAStack.clear();
        return Aloc;
        
        
        
    }

    Tensor getlocalVector(TTTensor y){
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
    
    TTTensor builtTTTensor(Tensor y){
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
            auto tmpxl=xbasis[d-1].get_component(pos);
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
    
    
    
    
    
    
};
