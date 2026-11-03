# Classification evaluation planning

`plan_classification_evaluation` creates validated assignments over an existing
`ClassificationDataset`. `validate_classification_evaluation` checks a saved plan
against current cases, metadata and the splitter runtime before execution.

Profiles support within-session whole-trial stratification, replay of saved
within-session holdouts, held-out participants and held-out sessions. Planning
uses core dependencies and does not read signal values or fit models.

See the [profile guide](../../docs/evaluation-profiles.md) for configuration,
identity fields, source-support requirements and the separate scoring contract.
