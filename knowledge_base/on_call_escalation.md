# On-Call Escalation Policy

## Severity levels

- **SEV1 (critical outage):** Page the primary on-call engineer immediately via PagerDuty. If there is no acknowledgment within 5 minutes, escalate to the secondary on-call engineer. If there is still no acknowledgment after 10 minutes, escalate to the engineering manager on the escalation roster.
- **SEV2 (degraded service):** Page the primary on-call engineer. Escalate to secondary after 15 minutes without acknowledgment.
- **SEV3 (minor issue):** Create a ticket in the incident tracker. No paging required; the on-call engineer picks it up during business hours.

## Rotation

On-call rotations run weekly, Monday 9:00 AM to the following Monday 9:00 AM, in the on-call engineer's local timezone. The rotation schedule is maintained in PagerDuty under the "Platform Engineering" schedule.

## Escalation contacts

1. Primary on-call engineer (PagerDuty)
2. Secondary on-call engineer (PagerDuty)
3. Engineering manager (Slack: #eng-leadership, or phone for SEV1)
4. VP of Engineering (SEV1 only, after 30 minutes without resolution)

## Post-incident

Every SEV1 and SEV2 incident requires a postmortem document within 48 hours of resolution, following the standard postmortem template.
