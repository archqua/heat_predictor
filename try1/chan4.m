for i = 1:4
  printf("%d\n",i)
  chan_fname = sprintf("channel%d.csv", i-1)
  nopen_fname = sprintf("new%dstiffoffset", i-1)
  pen_fname = sprintf("new%dstiffpenaltyoffset", i-1)
  chan_data = csvread(chan_fname)(2:end,:);
  nopen_data = csvread(nopen_fname);
  pen_data = csvread(pen_fname);
  xc = chan_data(1:end,1); yc = chan_data(1:end,2);
  xnp = nopen_data(1:end,1); ynp = nopen_data(1:end,2);
  xp = pen_data(1:end,1); yp = pen_data(1:end,2);
  figure(i);
  clf;
  plot(xc, yc, xnp, ynp, xp, yp)
  grid on;
  legend(['actual'; 'stiff'; 'penalty'])
  title(sprintf("channel %d", i-1))
endfor;
