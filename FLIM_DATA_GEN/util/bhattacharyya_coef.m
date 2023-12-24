function d = bhattacharyya_coef(X1, X2)
X1=histogram(X1);
X2=histogram(X2);
BC=sum(sqrt(X1.Values*X2.Values));
d=-log(BC);
end