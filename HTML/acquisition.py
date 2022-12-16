import time
import matplotlib.pyplot as plt

from bitalino import BITalino

# The macAddress variable on Windows can be "XX:XX:XX:XX:XX:XX" or "COMX"
# while on Mac OS can be "/dev/tty.BITalino-XX-XX-DevB" for devices ending with the last 4 digits of the MAC address or "/dev/tty.BITalino-DevB" for the remaining
macAddress = "20:18:05:28:73:33"

# This example will collect data for 5 sec.
running_time = 5

batteryThreshold = 30
acqChannels = [0]
samplingRate = 1000
nSamples = 5000
digitalOutput_on = [1, 1]
digitalOutput_off = [0, 0]

# Connect to BITalino
device = BITalino(macAddress)

# Set battery threshold
device.battery(batteryThreshold)

# Read BITalino version
print(device.version())

# Start Acquisition
device.start(samplingRate, acqChannels)

sample = []

start = time.time()
end = time.time()
while (end - start) < running_time:
    # Read samples
    sample = device.read(nSamples)
    print(sample[:, 5])
    #print(device.read(nSamples))
    end = time.time()

# Turn BITalino led and buzzer on
#device.trigger(digitalOutput_on)

#print(sample)
plt.plot(sample[:, 5])
#plt.ylim(0, 1024)

# Script sleeps for n seconds
time.sleep(running_time)

# Turn BITalino led and buzzer off
#device.trigger(digitalOutput_off)

# Stop acquisition
device.stop()

# Close connection
device.close()

plt.show()