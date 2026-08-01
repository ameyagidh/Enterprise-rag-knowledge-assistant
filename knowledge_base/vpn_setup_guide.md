# VPN Setup Guide

## Prerequisites

- A company-issued laptop enrolled in the MDM (mobile device management) system.
- Your corporate SSO credentials (username + password + MFA device).
- The GlobalProtect VPN client, available from the internal Software Center.

## Installation steps

1. Open the internal Software Center and install "GlobalProtect VPN Client".
2. Launch GlobalProtect and enter the portal address: `vpn.internal.example.com`.
3. Sign in with your corporate SSO credentials when prompted.
4. Approve the MFA push notification on your registered device.
5. Once connected, you should see a green "Connected" indicator in the GlobalProtect menu bar icon.

## Troubleshooting

- **"Portal unreachable" error:** Confirm you have an active internet connection, then retry. If it persists, check the IT status page for known VPN outages.
- **MFA push not received:** Confirm your registered device has a working data or Wi-Fi connection. You can also request a backup code from the IT help desk.
- **Repeated disconnections:** Switch your GlobalProtect connection method from "UDP" to "TCP" in client settings, which is more reliable on some corporate and hotel networks.

## Support

For VPN issues not covered here, contact the IT help desk via Slack (#it-help) or email (it-support@example.com). Include your device name and the exact error message.
