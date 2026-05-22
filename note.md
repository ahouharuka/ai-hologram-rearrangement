傅里叶变换：$g(u,v) = \iint f(x,y)e^{i2\pi (ux+vy)}dudv$

设$f$单位是$\mathrm{A}$，$x,y$的单位是$\mathrm{p}$，那么$u,v$的单位是$\mathrm{p}^{-2}$，$g$的单位是$\mathrm{A}\cdot\mathrm{p}^{-2}$



$F_{focal}(x,y) = \iint F_{SLM}(u,v)e^{i2\pi (xu+yv)}dxdy$



假设SLM平面即波数域图像是$A_{SLM}(u,v)e^{i\varphi_{SLM}(u,v)}$，焦平面即空间域图像是$A_{focal}(x,y)e^{i\varphi_{focal}(x,y)}$

也就是说一共有这四个物理量：$A_{SLM},\varphi_{SLM},A_{focal},\varphi_{focal}$

那么这个物理背景问题（不管GS算法还是AI算法）是已知这四个物理量中的哪些，不知道哪些，我们需要求的是哪几个？（以及是否还有可以随意假设不会影响结论的，请讲明白）