% ----- PURPOSE ------

% This example tracks particles that have undergone simulated diffusion, 
% translation, stretch, or shear.


% ----- NOTE ------

% Tracker used in this example will be MUCH faster and more 
% robust when the assignmentoptimal function is compiled. From the 
% directory with the function, run;
%
%     mex assignmentoptimal.c
% 
% and it should generate a fast compiled binary for you. You can check 
% that matlab is running the compiled version by doing
% 
%     which assignmentoptimal
% 
% and checking that the result is a compiled file and not a .m file.


% ----- DATA FILES -----

% xyt_diff.mat, xyt_trans.mat, xyt_stretch.mat, xyt_shear.mat, xyt_tfm.mat
% contain an array, xyt, of particle location and times.

% Structure of xyt: 
% xyt          (N+M)x3 array where N is the number of
                        % particles in the first frame and M is the number 
                        % of particles in the second frame.
                        % xyt(i,1) is the x-coordinate 
                        % xyt(i,2) is the y-coordinate
                        % xyt(i,3) is the image number
                       
                       
% The tracking parameters chosen below are as follows.  All of them are
% required for the strain-tracker. Only the first two are required for the
% diffusion tracker.

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


% ----- OUTPUTS -----

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
%                       In cases of the diffusion trackr, the structure is set 
%                       to NaN.

%%  Load Data

clear
close all

data = 'stretch'; % choose 'diff', 'stretch', 'shear', 'trans', or 'tfm'

switch data
    case 'diff'
        load xyt_diff  %load variable 'xyt'
        param.maxdisp = 6;  param.mode = 1;
        
    case 'stretch'
        load xyt_stretch  %load variable 'xyt'
        param.minn = 3;  param.rmax=100; param.maxdisp = 35;
        param.maxstr = 0.2; param.maxcompr = 0;  param.mode = 0;
    case 'shear'
        load xyt_shear;  %load variable 'xyt'
        param.minn = 3;  param.rmax=100; param.maxdisp = 25;
        param.maxstr = 0.2; param.maxcompr = 0.2;  param.mode = 0;
    case 'trans'
        load xyt_trans; %load variable 'xyt'
        param.minn = 2;  param.rmax=100;  param.maxdisp = 20;
        param.maxstr = 0.001; param.maxcompr = 0.001;  param.mode = 0;
    case 'tfm'
        load xyt_tfm; %load variable 'xyt'
        param.minn = 2;  param.rmax=100;  param.maxdisp = 22;
        param.maxstr = 0.5; param.maxcompr = 0.5; param.mode = 0;
end

%% analyze the data

set(0,'RecursionLimit',800)         %  Sets the recursion limit to 800.  
                                    % As of 08.2016 computers may crash for 
                                    % numbers much higher than 800.

% The following runs the tracking function:
[cm, tks, str] = Tracker(xyt,param);  

d = trck2dsp(tks);  % this function reformats tks into a more 
                      % convenient structure as follows:

% d contains all positions and displacements for each time point.  
% d(t).r      % 2xN matrix of particle positions at the first time
                 % point, where N is the number of particles
% d(t).dr:    % 2xN matrix of displacements at time t relative to t=1.

ts = unique(xyt(:,3));  % finding time points

ts1 = min(ts);  % first time point
ts2 = max(ts);  % second time point

ind1 = find(xyt(:,3)==ts1);  % indices of xyt_diff that correspond to 
                           % particle in the first frame
ind2 = find(xyt(:,3)==ts2);  % indices of xyt_diff that correspond to 
                           % particle in the second frame

%% plot results
plot(xyt(ind1,1), xyt(ind1,2), 'ok', 'MarkerSize', 7)  % Plotting particle 
                                                       % positions in the 
                                                       % first frame 
                                                       % with 'o' markers
hold on
plot(xyt(ind2,1), xyt(ind2,2), 'xk', 'MarkerSize', 7)  % Plotting particle 
                                                       % positions in the 
                                                       % second frame 
                                                       % with 'x' markers

set(gca,'FontSize',16,'Color', 'w')

axis off

scl = 1;  % scale for enlarging arrows representing particle displacements

% Plotting arrows from particles in the first frame to the corresponding 
% particles in the second frame 
quiver(d(1).r(:,1),d(1).r(:,2),scl*d(2).dr(:,1),scl*d(2).dr(:,2),0,'r')