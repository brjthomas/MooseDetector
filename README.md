# MooseDetector
An edge AI project that runs live inference on thermal images. 

## System Requirements 🔧
- Raspberry Pi 5 (aarch64)
- Raspberry Pi OS (64-bit)
- Seek Thermal camera
- Hailo-8L AI HAT (optional for now)

## External Dependencies 📦

### Seek Thermal SDK (Python)

This project uses the Seek Thermal Python SDK in editable mode.

Once downloaded, the SDK is located at:
`~/projects/ThermalCameraSDK/seekcamera-python`

The virtual environment links to this path using a `.pth` file, allowing
the `seekcamera` module to be imported without copying it into site-packages.

To (re)install:
```bash
source venv/bin/activate
pip install -e ~/projects/ThermalCameraSDK/seekcamera-python
```

### Seek Thermal SDK Install Instructions (C)
1. Download Seek Thermal SDK v4.4.2.20 from the [Seek Developer Portal](https://developer.thermal.com/support/home)
2. Extract to
```bash
~/projects/ThermalCameraSDK/
```
3. Install SDL support
```bash
sudo apt-get install libsdl2-dev
```
4. In the SDK folder
```bash
cd ~/projects/ThermalCameraSDK/Seek_Thermal_SDK_4.4.2.20/aarch64-linux-gnu/
sudo cp driver/udev/10-seekthermal.rules /etc/udev/rules.d
sudo udevadm control --reload
```
5. Set the environment variable
```bash
echo 'export LD_LIBRARY_PATH=~/projects/ThermalCameraSDK/Seek_Thermal_SDK_4.4.2.20/aarch64-linux-gnu/lib:$LD_LIBRARY_PATH' >> ~/.bashrc
source ~/.bashrc
```
### Setup Virtual Environement
```bash
cd ~/projects/MooseDetector/
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### Verify Installation 
```bash
python -c "import seekcamera; print('Seek SDK loaded successfully')"
```

## Development Setup 🛠️

This project is developed directly on a Raspberry Pi 5 and accessed remotely from a development machine using SSH and VS Code. The steps below describe how the development environment is configured.

### Network Configuration (Static IP)

To ensure reliable remote access, the Raspberry Pi is configured with a static IP address on the local network. This prevents the IP from changing between reboots and simplifies SSH and VS Code connections.

Configure Static IP (Raspberry Pi OS)

Edit the DHCP configuration file:
```bash
sudo nano /etc/dhcpcd.conf
```
Add the following at the bottom (adjust values to match your network):
```bash
interface wlan0
static ip_address=192.168.5.44/24
static routers=192.168.5.1
static domain_name_servers=192.168.5.1 8.8.8.8
```
Restart networking (or reboot):
```bash
sudo reboot
```
Verify the IP address:
```bash
ip a
```
### Enable SSH Access
SSH is enabled on the Raspberry Pi to allow remote terminal access.
```bash
sudo systemctl enable ssh
sudo systemctl start ssh
```
Verify SSH is running:
```bash
systemctl status ssh
```
From the development machine, connect using:
```bash
ssh moose@192.168.1.50
```
(Replace moose and the IP address as needed.)
### VS Code Remote SSH Setup 💻
Development is done using VS Code Remote – SSH, allowing code editing, debugging, and terminal access directly on the Raspberry Pi.

On the development machine
- Open VS Code
- Install the Remote – SSH extension
Edit (or create) the SSH config file on the development machine:
- Open Command Palette (Ctrl+Shift+P)
- Select Remote-SSH: Open SSH configuration file
- Add:
```bash
Host moosedetector-pi
    HostName 192.168.1.50
    User moose
```
Connect from VS Code
- Open Command Palette (Ctrl+Shift+P)
- Select Remote-SSH: Connect to Host
- Choose moosedetector-pi

VS Code will open a new window connected directly to the Raspberry Pi, with full access to:
- The project directory
- Python virtual environments
- Terminals
- Debugger
- Git
