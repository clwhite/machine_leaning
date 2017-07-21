import csv

DATA = {1: {'admin.': 0,
            'blue-collar': 1,
            'entrepreneur': 2,
            'housemaid': 3,
            'management': 4,
            'retired': 5,
            'self-employed': 6,
            'services': 7,
            'student': 8,
            'technician': 9,
            'unemployed': 10,
            'unknown': 11},
        2: {'divorced': 0,
            'married': 1,
            'single': 2,
            'unknown': 3},
        3: {'primary': 0,
            'secondary': 1,
            'tertiary': 2,
            'unknown': 3},
        4: {'no': 0,
            'yes': 1},
        6: {'no': 0,
            'yes': 1},
        7: {'no': 0,
            'yes': 1},
        8: {'unknown': 0,
            'cellular': 1,
            'telephone': 2},
        10: {'jan': 1,
             'feb': 2,
             'mar': 3,
             'apr': 4,
             'may': 5,
             'jun': 6,
             'jul': 7,
             'aug': 8,
             'sep': 9,
             'oct': 10,
             'nov': 11,
             'dec': 12},
        15: {'unknown': 0,
             'failure': 1,
             'success': 2,
             'other': 3},
        16: {'no': 0,
             'yes': 1}}


def convert(src, trg):
  count = 0
  with open(src, 'r') as rf:
    rc = csv.reader(rf, delimiter=';')
    with open(trg, 'w') as wf:
      wc = csv.writer(wf, delimiter=',')  # delim is for tensorflow
      rc.next()  # remove first
      for l in rc:
        for index, d in DATA.items():
          key = l[index].strip()
          l[index] = d[key]
        wc.writerow(l)
        count += 1
  with open(trg, 'r') as rf:
    rc = csv.reader(rf, delimiter=';')
    for l in rc:
      print l

  print "count:%s" % count

if __name__ == '__main__':
  import sys
  print sys.argv
  if len(sys.argv) < 3:
    print "input source target"
    sys.exit()
  src = sys.argv[1]
  trg = sys.argv[2]
  convert(src, trg)
