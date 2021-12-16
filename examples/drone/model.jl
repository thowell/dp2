function double_integrator_1D(h, y, x, u, w) 
    A = [1.0 h; 
         0.0 1.0] 
    B = [0.0; 1.0]

    y - (A * x + B * u[1])
end

function double_integrator_2D(h, y, x, u, w) 
    A = [1.0 0.0 h   0.0; 
         0.0 1.0 0.0 h; 
         0.0 0.0 1.0 0.0;
         0.0 0.0 0.0 1.0]

    B = [0.0 0.0; 
         0.0 0.0; 
         1.0 0.0; 
         0.0 1.0]

    y - (A * x + B * u)
end

function double_integrator_2D_forward(h, x, u, w) 
     A = [1.0 0.0 h   0.0; 
          0.0 1.0 0.0 h; 
          0.0 0.0 1.0 0.0;
          0.0 0.0 0.0 1.0]
 
     B = [0.0 0.0; 
          0.0 0.0; 
          1.0 0.0; 
          0.0 1.0]
 
     A * x + B * u
 end