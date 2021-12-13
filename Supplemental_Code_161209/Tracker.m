function [cm_tot, tks, str] = Tracker(xyt,param)
% -------------------------------------------------------------------------
%
% PURPOSE:  
% This function assigns particle identifications in 2 frames based on 
% either:
%       -  minimization of squared displacements (diffusion tracker)
%       -  minimal energy storage/dissipation (strain tracker)
%
% INPUTS: 
% param.mode;       --  set to 0 for strain tracker and 1 for 
%                       diffusion tracker.
% param.maxdisp;    --  max displacement of a particle one expects from
%                       image to image.  Make this smaller, and the code
%                       will run faster, but you might miss some pairs.
% param.maxstr;     --  maximum expected stretch - used for (1) estimating
%                       maximum change in separation of two
%                       neighboring particles and (2) for eliminating
%                       candidate particle pairs that give 
%                       strain > param.maxstr. This is a positive number
%                       usually < 1.
% param.maxcompr;   --  maximum expected compression. Used to eliminate
%                       candidate particle pairs that give 
%                       strain < - param.maxcompr. This is a positive  
%                       number usually < 1. If in doubt, set
%                       param.maxcompr = param.maxstr.
% param.minn;       --  minimum number of neighbors a particle needs to
%                       have in order to calculate a local strain.  In this 
%                       2D code, minn must be at least 2.  With a larger 
%                       value you will have more accurate strain                         
%                       measurements, but you may miss some particles.                       
% param.rmax;       --  radius around a given particle to look for
%                       neighbors. This should be chosen so you typically 
%                       have 3-6 neighbors per particle. if this value is
%                       too large, you will underestimate strain gradients
%
% OUTPUTS: 
% cm                --  Cost matrix associated with the found particle 
%                       identification: cm(i,j) is the cost associated with
%                       assigning particle i in the first timepoint to
%                       particle j in the second timepoint 
% tks               -- 4xN array where N is the number of particles such
%                       that tks(i,1) is the x-coordinate of a given
%                       particle; tks(i,2) is the y-coordinate of a given
%                       particle; tks(i,3) is the image number, and
%                       tks(i,4) is the particle ID number.
% str               -- In cases of material deformation it is a 5xN array 
%                       such that str(i,1) is the xx strain of the
%                       i^th particle, str(i,2) is the yy strain of the 
%                       i^th particle, str(i,3) is the xy strain of the
%                       i^th particle, and str(i,4) is the yx strain of the
%                       i^th partice. str(i,5) is the ID of the i^th
%                       particle.
%                       In cases of Brownian motion, the structure is set 
%                       to NaN.
%
%
% NOTE: 
% This function relies on the assignmentoptimal function by Markus Buehren.
% 
%   <a href="assignment.html">assignment.html</a>  <a href="http://www.mathworks.com/matlabcentral/fileexchange/6543">File Exchange</a>  <a href="https://www.paypal.com/cgi-bin/webscr?cmd=_s-xclick&hosted_button_id=EVW2A4G2HBVAU">Donate via PayPal</a>
% 
% For your convenience we have bundled the 2014 version with this code,
% but more recent versions may be available online.
%
% This function will be faster and more robust when the 
% assignmentoptimal function is compiled. From the directory with the
% function, run;
%
%     mex assignmentoptimal.c
% 
% and it should generate a fast compiled binary for you. You can check 
% that matlab is running the compiled version by doing
% 
%     which assignmentoptimal
% 
% and checking that the result is a compiled file and not a .m file.
%  
% KNOW ISSUES:
%   This code is written for 2D positions and displacements, but could be
%   easily extended to 3D.
%   This cose is written for two timepoints, but could be easily extended
%   to multiple timepoints.
%
% MODIFICATION HISTORY:
%   Written by Rostislav 'Ross' Boltyanskiy and Eric R. Dufresne, 
%   2015-2016.
% -------------------------------------------------------------------------


% Recasting input parameters
dif = param.mode;      
maxdisp = param.maxdisp;   
         

% finding time stamps
ts = unique(xyt(:,3));  
if length(ts)~=2
   error('xyt must include 2 time points!');   % this code only works with 
end                                            % two time points
ts1 = min(ts);  % first time point
ts2 = max(ts);  % second time point

%extract the initial and final particle positions
x1 = xyt(find(xyt(:,3)==ts1),1:2);    % particles positons in the 1st image
x2 = xyt(find(xyt(:,3)==ts2),1:2);    % particles positons in the 2nd image
np1 = length(x1);           % number of particles in the first image
np2 = length(x2);           % number of particles in the second image

%   calculate the square distances between all cross-time particle pairs 
    sdm = square_distance_matrix(x1,x2);    % matrix with the i-j entry 
                                            % equal to the square distances 
                                            % between the ith particle in 
                                            % the first frame and the jth 
                                            % particle iin the second frame  
%   make the diffusion cost matrix
    cm_diff = sdm;                         
    cm_diff(sdm > maxdisp^2) = Inf;         % forbid assignment with 
                                            % inter-particle distances 
                                            % larger than maxdisp

%   make the strain cost matrix
    cm = zeros(np1,np2);    % initializing cost matrix 
    cm(sdm > maxdisp^2) = Inf;      % Forbid matches with inter-particle 
                                    % distances larger than maxdisp
    
if dif == 0
% If we are in material deformation mode loop through all cross-time 
% particle pairs to calculate the cost matrix.

% recast other input parameters
maxstr = param.maxstr;  
maxcompr = param.maxcompr;
minn = param.minn;          
rmax = param.rmax; 

% Finding nearest neighbors at each time point. 
% The findnn function is defined as a subroutine below.
d1=findnn(x1,rmax);  
d2=findnn(x2,rmax);

for i = 1:np1
    for j = 1:np2
        
        %calculate the cost matrix
            if isfinite(cm(i,j))
                nnni = length(d1(i).nn);    % number of nearest neighbors 
                                            % of the i^th particle in the 
                                            % first frame
                nnnf = length(d2(j).nn);    % number of nearest neighbors 
                                            % of the j^th partcile in the 
                                            % second frame
                    if nnni > minn && nnnf > minn   
                            xn1 = [];       % this will be a list of 
                                            % nearest neighbors of particle 
                                            % i in the first frame
                            xn2 = [];       % this will be a list of 
                                            % nearest neighbors of particle 
                                            % j in the second frame 
                            for n = 1:nnni
                                xn1 = [xn1; x1(d1(i).nn(n),:)];  % nnni x 2 
                                                                 % matrix
                            end
                            
                            for n = 1:nnnf
                                xn2 = [xn2; x2(d2(j).nn(n),:)]; % nnnf by 2 
                                                                % matrix
                            end
                            
                            xi1 = x1(i,:);      % coordinate of particle i 
                                                % in the first frame 
                            xj2 = x2(j,:);      % coordinate of particle j 
                                                % in the second frame
                            
                             % Find strain associated with particle 
                             % pairing.  
                             % The subroutine choosenn2track is defined
                             % below.
                             % Returns the strain matrix unless there are
                             % not enough tracked neighbors, in which case
                             % it returns Inf
                            [eps, err] = calcstrain(xi1, xn1, xj2, xn2,... 
                            minn, maxstr, maxcompr); 
                            
                           % Setting the cost of particle pairings
                                                                                           
                            if eps == Inf       
                                cm(i,j) = Inf;

                            else
                                % calcute the strain energy / energy
                                % dissipation rate
                                cm(i,j) = trace(eps^2);
                                %cm(i,j) = sum(eigs(eps).^2);
                                Strain(i,j).eps = eps;                        
                            end
                    else
                        cm(i,j) = Inf;      % set cost to infinity if a 
                                            % particle doesn't have enough 
                                            % neighbors
                        Strain(i,j).eps = NaN;  
                    end

            end
    end
end
    
cm_tot = cm;

elseif dif == 1;  
    % If we are in Brownian motion mode, set cost based on square 
    % displacements
    cm_tot = cm_diff;   
end


% Running the Hungarian algorith on the cost matrix

assignment = assignmentoptimal(cm_tot);

% Format the output of the Hungarian method
tks=assign2trk(x1,x2,assignment); % The assign2trk function is embedded in 
                                  % this main function

% Format the strain if in material deformation mode.

if dif == 1;
    str = NaN;
else
    str=assign2str(Strain,assignment);
end

end

% --------------------------------------------------------------------


function d = findnn(x,rmax)

% This function finds the nearest neighbors of particles within a region,
% rmax
%
% INPUTS
% x           -- 2xN list of N particle positions
% rmax        -- radius around each particle within which neighbors are 
%                found

% OUTPUTS
% d           -- list of nighbors such that d(i).nn(n) is the index of the
                 % n^th neighbor of the i^th particle.  Then
                 % x(di(i).nn(n),:) is the coordinate of the n^th neighbor.

% Recast input parameter
xf=x;

% find the nearest neighbors for the final frame
trif=delaunay(xf(:,1),xf(:,2));

for n=1:length(xf)
%find ith particle and jth vertex for all triangles that include particle n
[i,j]=ind2sub(size(trif),find(trif==n));

 nt=length(i);  %number of triangles involving the nth particle
        vts = [];
        for k=1:length(i)  %loop through all triangles involving nth 
                           % particle
            vts=[vts,trif(i(k),:)];     % make a list of all paricles in 
                                        % triangles with nth particle
        end
        vts=setdiff(unique(vts),n);     % get rid of n and all duplicates
        df(n).nn=vts;  %now we have a  list of candidate nearest neighbors
        %list of distance of each possible nearest neighbors from our
        %particle
        r=sqrt((xf(vts,1)-xf(n,1)).^2+(xf(vts,2)-xf(n,2)).^2);
        ind = find(r<rmax);  %throw away ones that are too far away.
        df(n).nn=vts(ind);              
        d=df;              
end
end


% -------------------------------------------------------------------


function [eps, d2] = calcstrain(xi1, xn1, xj2, xn2, minn, maxstr, maxcompr)
% This functions matches up two candidate particles and calculates the
% strain associated with that partcile pair by alligning the
% particles' neighbors.

%INPUTS
% xi1 = initial particle position, 2-component vector
% xj2 = final candidate particle position, 2-component vector
% xn1 = coordinates of initial particle neightbors: 2xN where N is the 
%       number of neighbors
% xn2 = coordinates  of final candidate particle neightbors: 2xM where M is 
%       the number of neighbors
% minn = minimum number of neighbors that need to be tracked for us to
%           include the particle pair in the cost matrix
% maxstr = max strain from stretch we expect to estimate maxdisp.
% maxcompr = max strain from compression we expect to estimate maxdisp.

nni = length(xn1);   % number of neighbors of particle in the initial frame
nnf = length(xn2);   % number of neighbors of particle in the final frame

% Initializing shifted neighbor coordinates
xn1s = zeros(size(xn1));    
xn2s = zeros(size(xn2));

% Compiling new neighbor coordinates centered at the particle of interest
xn1s(:,1) = xn1(:,1) - xi1(1);
xn1s(:,2) = xn1(:,2) - xi1(2);
xn2s(:,1) = xn2(:,1) - xj2(1);
xn2s(:,2) = xn2(:,2) - xj2(2);

% Tracking the neighbors of candidate particles
% (you can't calculate the strain if you don't know who is who.  here we
% use the squared displacement tracker because the displacements are small 
% compared to the separations of neighboring particles.  This works very 
% well as long as there are not huge strain gradients.

    dn = pdist(xn1); % distances between particles in first frames
    mdn = max(dn);

    md = maxstr* mdn;  % maximum distance a neighbor can be expected to 
                       % move relative to other neighbors.

    sdm_nb = square_distance_matrix(xn1s,xn2s); % square distance matrix 
                                                % for neighbors   
    cm_nb = sdm_nb;                             % cost matrix for neighbors 
    cm_nb(sdm_nb > md^2) = Inf;                 % setting cost for 
                                                % neighbors that are too 
                                                % far away to infinity 
    cm_nb(sdm_nb < 0) = Inf;
    
    assign_nb = assignmentoptimal(cm_nb);      % running Hungarian on cm_nb

    good_a = [];                                % initializing matrix of 
                                                % neighbors that were 
                                                % tracked

    ntr = length(assign_nb(assign_nb>0));       % number of neighbors that 
                                                % were tracked

if ntr > minn

    for i=1:length(assign_nb)               % filling in matrix identifying 
                                            % neighbors that were tracked
       if assign_nb(i) > 0                  
                                            % good_a(i,1) = tracked 
                                            % neighbor at the first time 
                                            % point
       good_a = [good_a; i assign_nb(i)];   % good_a(i,2) = corresponding 
                                            % tracked neighbor at the 
                                            % second time point
       end 
    end
     
    % initializing coordinates of tracked neighbors
    xn1t = [];  
    xn2t = []; 
    
    % Sort the neighbor positions so that tracked pairs are in the same row
        
     for i = 1:length(good_a)
             xn1t = [xn1t; xn1s(good_a(i,1),1:2) + [xi1(1) xi1(2)]];
             xn2t = [xn2t; xn2s(good_a(i,2),1:2) + [xj2(1) xj2(2)]];         
     end  

     % Calculating strain
     [eps, d2] = falkstrain(xi1, xn1t, xj2, xn2t);

     % Forbid strains that are larger than maxstr or smaller than -maxcompr  
     if  max(eig(eps)) > maxstr || min(eig(eps)) < -maxcompr
        eps = Inf;
     end
     
     
else
    eps = Inf;
    d2 = Inf;
end


end



% -------------------------------------------------------------------


function tks=assign2trk(xi,xf,assignment)

% This function reformats the particle pairing from the minimal_assignment 
% function

%INPUTS
% xi                -- particle positions in the first image
% xf                -- particle positions in the second image
% assingment        -- output of assignmentoptimal function

%OUTPUTS
% tks               -- 4xN structure where N is the number of particles
                        % such that tks(i,1) is the x-coordinate of
                        % a given particle; tks(i,2) is the y-coordinate of
                        % a given particle; tks(i,3) is the image number,
                        % and tks(i,4) is the particle ID number.

% initializing tks structure                  
tks = [];

%initializing particle ID
id = 0;

for i = 1:length(assignment)
    if assignment(i) ~= 0
    id = id+1;
    tks = [tks; xi(i,1), xi(i,2), 1, id; xf(assignment(i),1), ...
        xf(assignment(i),2), 2, id];
    end 
end

end


%----------------
function [ep,d2]=falkstrain(roi,rni,rof,rnf)
% [ep,d2]=falkstrain(roi,rni,rof,rnf)
% calculates best estimate of strain and D^2 (estimate of non-affine 
% transformation) according to falk and langer pre1998, Eqs 2.11-2.14
% INPUTS
% roi, rof: [1,d] row vectors of initial and final positions for particle
% of interest
% rni, rnf: [nn,d] matrices of initial and final positions for nearest
% neighbors of particles of interest
%OUTPUTS
%ep: ep(i,j) is the ith and jth elements of the best estimate for the local
%strain tensor.  NOTE: in falk and langer this is not symmetric.  I
%only return the symmetric part of this tensor (as alluded to by Schall,
%Weitz and Spaepen, Nature 2006).
%d2: a single number capturing the amount of non-affine deformation
%CREATED BY:
% Eric R. Dufresne, Yale University June 18 2009

[nn,d]=size(rni);

%calculate Falk's X and Y
X=zeros(d);Y=zeros(d);
for i=1:d
    for j=1:d
        for n = 1:nn
            X(i,j)=X(i,j) + (rnf(n,i)-rof(i))*(rni(n,j)-roi(j));
            Y(i,j)=Y(i,j) + (rni(n,i)-roi(i))*(rni(n,j)-roi(j));
        end
    end
end

%Calculate Falk's strain
invY=inv(Y);
ep=zeros(d);
for i=1:d
    for j=1:d
        for k=1:d
            ep(i,j)=ep(i,j)+ X(i,k)*invY(j,k);
        end
    end
end
id = eye(d);
ep=ep-id;

%calculate D^2
d2=0;
for n=1:nn
    for i=1:d
        tmp=0;
        for j=1:d
            tmp = tmp + (id(i,j)+ep(i,j))*(rni(n,j)-roi(j));
           % display(['n ',int2str(n),' i ',int2str(i),' j ' int2str(j)]);
            %tmp;
        end
        d2=d2+((rnf(n,i)-rof(i)) - tmp )^2;
    end
end
%divide D^2 by number or nearest neigbors
d2=d2/nn;
%normalize D^2 by nearest neigbhor distance
for n=1:nn
    r2(n)=sum((rni(n,:)-roi).^2);
end
mnr2=(mean(r2));
d2=d2/mnr2;

%return the symmetrized strain tensor
ep = .5*(ep+ep');

end


function str = assign2str(Strain,assignment)
% This function assigns a strain to each particle in cases of material
% deformation.
%
% INPUTS:
% Strain        -- NxM matrix where N is the number of particles in the 
%                   first frame and M is the number of particles in the 
%                   second frame
%                   In cases of stretch and shear, Strain(i,j).eps is a 2x2
%                   symmetrized strain tensor.  This is the strain
%                   associated with pairing particle i in the first frame
%                   with particle j in the second frame.
%                   In cases of diffusion, Strain(i,j).eps = NaN.
%
% assignment    -- Output of the assignmentoptimal function. It is a
%                   structure such that if assignment(m,1) = n, then 
%                   particle m in the first frame is identified as particle
%                   n in the second frame.                   
% 
% OUTPUTS:
% str           -- Nx5 structure such that str(i,1) = is the xx strain
%                   associated with particle i in the first frame.
%                   str(i,2) = is the yy strain associated with particle i 
%                   in the first frame.  str(i,3) = is the xy strain 
%                   associated with particle i in the first frame.  
%                   str(i,4) is the yx strain associated with particle i in     
%                   the first frame. str(i,5) is the id of the i^th
%                   particle

% initializing strain structure
str = [];

% initializing particle id
id = 0;
% formatting strain based on the structures Strain and assignment

for i = 1:length(assignment)
    if assignment(i,1) ~= 0;
        id = id+1;
        exx = Strain(i,assignment(i,1)).eps(1,1);
        eyy = Strain(i,assignment(i,1)).eps(2,2);
        exy = Strain(i,assignment(i,1)).eps(1,2);
        eyx = Strain(i,assignment(i,1)).eps(2,1); 
    
        str = [str; exx, eyy, exy, eyx, id]; 
    end
end

end


function d = square_distance_matrix(a,b)
% square_distance_matrix - computes Euclidean distance-squared matrix
%
% E = distance(A,B)
%
%    A - (DxM) matrix 
%    B - (DxN) matrix
%
% Returns:
%    E - (MxN) Euclidean distances squared between vectors in A and B
%
%
% Description : 
%    This fully vectorized (VERY FAST!) m-file computes the 
%    Euclidean distances squared between two vectors by:
%
%                 ||A-B|| = sqrt ( ||A||^2 + ||B||^2 - 2*A.B )
%
% Example : 
%    A = rand(100,400); B = rand(200,400);
%    d = distance(A,B);

% Author   : Roland Bunschoten
%            University of Amsterdam
%            Intelligent Autonomous Systems (IAS) group
%            Kruislaan 403  1098 SJ Amsterdam
%            tel.(+31)20-5257524
%            bunschot@wins.uva.nl
% Last Rev : Oct 29 16:35:48 MET DST 1999
% Tested   : PC Matlab v5.2 and Solaris Matlab v5.3
% Thanx    : Nikos Vlassis
% Modified : Jason Merrill 5 February 2008

% Copyright notice: You are free to modify, extend and distribute 
%    this code granted that the author of the original code is 
%    mentioned as the original author of the code.

if (nargin ~= 2)
   error('Not enough input arguments');
end

if (size(a,2) ~= size(b,2))
   error('A and B should be of same dimensionality');
end

d = bsxfun(@plus,sum(a.*a,2),sum(b.*b,2)') - 2*a*b';

end
