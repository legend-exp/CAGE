# https://matplotlib.org/users/customizing.html

#figure.titlesize : small  # size of the figure title (Figure.suptitle())
figure.figsize   : 8, 5    # figure size in inches
#figure.dpi       : 200    # figure dots per inch
figure.facecolor : white
figure.edgecolor : white
figure.autolayout : False   # Automatically apply 'plt.tight_layout'

lines.linewidth : 3
lines.markersize : 10
legend.fontsize : 18
legend.handlelength : 1
axes.titlesize : 17
axes.labelsize : 20
axes.linewidth : 1
axes.grid : True
grid.linestyle : :
xtick.labelsize : 18       # font size of tick labels
xtick.direction : in
ytick.direction : in
ytick.labelsize : 18
xtick.major.size : 10      # major tick size in points
xtick.minor.size : 5       # minor tick size in points
ytick.major.size : 10
ytick.minor.size : 5
xtick.minor.visible  : True   # turn minor ticks on by default
ytick.minor.visible  : True

# -- this font is close to phys rev
font.family : STIXGeneral
font.serif : STIX
mathtext.fontset : stix

## The figure subplot parameters.  All dimensions are a fraction of the figure width and height.

# this choice of parameters centers the plot box, so that it shows up properly centered in a publication.

#figure.subplot.left:   0.1  # the left side of the subplots of the figure
#figure.subplot.right:  0.9    # the right side of the subplots of the figure

figure.subplot.left: 0.12
figure.subplot.right: 0.975

figure.subplot.bottom: 0.12   # the bottom of the subplots of the figure
figure.subplot.top:    0.99   # the top of the subplots of the figure
#figure.subplot.wspace: 0.2    # the amount of width reserved for space between subplots,
                               # expressed as a fraction of the average axis width
#figure.subplot.hspace: 0.2    # the amount of height reserved for space between subplots,
                               # expressed as a fraction of the average axis height
